"""Sentinel-1 IW GRD 批量下载（Copernicus Data Space / ASF）。

凭据（不进版本库，见 .gitignore 的 .secrets/）：
    - CDSE：项目根目录 .secrets/credentials.json（{"cdse": {"username", "password"}}），
      或环境变量 CDSE_USER / CDSE_PASS；
    - ASF：环境变量 EARTHDATA_USER / EARTHDATA_PASS（ASF 走 Earthdata 登录）。

用法：
    python data_preprocessing/download/download_sentinel1.py \
        --source cdse --aoi "109,18,114,23" --start 2023-01-01 --end 2023-12-31 \
        --out data/raw/sentinel1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))
from utils.logger import get_logger

logger = get_logger("download_s1")

CDSE_TOKEN_URL = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
                  "/protocol/openid-connect/token")
CDSE_ODATA = "https://catalogue.dataspace.copernicus.eu/odata/v1"


def load_credentials(section: str) -> tuple[str, str] | None:
    """从 .secrets/credentials.json 读某一节的用户名密码；文件或节不存在返回 None。"""
    cred_file = PROJECT_ROOT / ".secrets" / "credentials.json"
    if cred_file.exists():
        creds = json.loads(cred_file.read_text(encoding="utf-8")).get(section)
        if creds:
            return creds["username"], creds["password"]
    return None


def _cdse_credentials() -> tuple[str, str]:
    user = os.environ.get("CDSE_USER")
    pwd = os.environ.get("CDSE_PASS")
    if not user or not pwd:
        file_creds = load_credentials("cdse")
        if file_creds:
            user = user or file_creds[0]
            pwd = pwd or file_creds[1]
    if not user or not pwd:
        logger.error("未找到 CDSE 凭据：请创建 .secrets/credentials.json "
                     "或设置 CDSE_USER / CDSE_PASS 环境变量")
        sys.exit(1)
    return user, pwd


def _cdse_token() -> str:
    import requests

    user, pwd = _cdse_credentials()
    r = requests.post(CDSE_TOKEN_URL, data={
        "grant_type": "password",
        "client_id": "cdse-public",
        "username": user,
        "password": pwd,
    }, timeout=60)
    r.raise_for_status()
    return r.json()["access_token"]


def _cdse_search(wkt: str, start: str, end: str, max_results: int):
    """OData 检索 Sentinel-1 IW GRD，返回 [{id, name, size, start}]。客户端再按
    文件名过滤 IW_GRDH，避免冗长的 Attributes 过滤表达式。"""
    import requests

    filt = ("Collection/Name eq 'SENTINEL-1' and "
            "OData.CSC.Intersects(area=geography'SRID=4326;" + wkt + "') and "
            f"ContentDate/Start ge {start}T00:00:00.000Z and "
            f"ContentDate/Start le {end}T23:59:59.999Z")
    r = requests.get(f"{CDSE_ODATA}/Products", params={
        "$filter": filt,
        "$top": max_results,
        "$orderby": "ContentDate/Start asc",
        "$select": "Id,Name,ContentLength,ContentDate",
    }, timeout=120)
    r.raise_for_status()
    raw = r.json().get("value", [])
    # 同一景的 COG 版与普通版只保留一个：按采集标识去重（去掉末尾校验码与 _COG 标记），
    # 默认保留经典 .SAFE（SNAP 兼容性更稳）
    def _acq_key(name: str) -> str:
        n = name[:-5] if name.endswith(".SAFE") else name
        if n.endswith("_COG"):
            n = n[:-4]
        return n.rsplit("_", 1)[0]

    best = {}
    for p in raw:
        name = p["Name"]
        if "_IW_GRDH_" not in name:  # 只要 IW 模式高分辨率 GRD
            continue
        key = _acq_key(name)
        if key in best and best[key]["Name"].endswith("_COG.SAFE") \
                and not name.endswith("_COG.SAFE"):
            best[key] = p  # 已有的是 COG 版，换成经典版
        else:
            best.setdefault(key, p)
    items = []
    for p in best.values():
        items.append({
            "id": p["Id"],
            "name": p["Name"],
            "size": p.get("ContentLength"),
            "start": p.get("ContentDate", {}).get("Start"),
        })
    return items


def _cdse_download(items, out: Path, token: str, max_retries: int = 10) -> None:
    import requests

    out.mkdir(parents=True, exist_ok=True)
    for i, it in enumerate(items, 1):
        dest = out / it["name"]  # OData $value 直接给 .zip
        if not dest.suffix:
            dest = dest.with_suffix(".zip")
        expected = it["size"] or 0
        if dest.exists() and expected and dest.stat().st_size == expected:
            logger.info("[%d/%d] 已存在跳过 %s", i, len(items), dest.name)
            continue
        logger.info("[%d/%d] 下载 %s (%.2f GB)", i, len(items), it["name"],
                    expected / 1e9)
        url = f"{CDSE_ODATA}/Products({it['id']})/$value"
        for attempt in range(1, max_retries + 1):
            pos = dest.stat().st_size if dest.exists() else 0
            headers = {"Authorization": f"Bearer {token}"}
            mode = "wb"
            if pos:  # 断点续传
                headers["Range"] = f"bytes={pos}-"
                mode = "ab"
            try:
                # 目录服务会 302 到 download.dataspace.copernicus.eu，requests
                # 跨域重定向会丢弃 Authorization 头，需手动跟随并重新附带头
                r0 = requests.get(url, headers=headers, stream=True, timeout=60,
                                  allow_redirects=False)
                final_url = url
                if r0.status_code in (301, 302, 303, 307, 308):
                    final_url = r0.headers["Location"]
                    r0.close()
                with requests.get(final_url, headers=headers, stream=True,
                                  timeout=600) as r:
                    r.raise_for_status()
                    if mode == "ab" and r.status_code != 206:
                        # 服务器不支持 Range：作废续传，整文件重下
                        logger.warning("服务器未接受断点续传（状态 %d），整文件重下",
                                       r.status_code)
                        mode = "wb"
                    with open(dest, mode) as fp:
                        for chunk in r.iter_content(1 << 20):
                            fp.write(chunk)
                break
            except (requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                got = dest.stat().st_size if dest.exists() else 0
                logger.warning("下载中断（%s），已下 %.2f/%.2f GB，第 %d/%d 次重试",
                               type(e).__name__, got / 1e9, expected / 1e9,
                               attempt, max_retries)
                if attempt == max_retries:
                    raise
        if expected and dest.stat().st_size != expected:
            raise IOError(f"{dest.name} 大小不符：{dest.stat().st_size} != {expected}")
    logger.info("下载完成 → %s", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["cdse", "asf"], default="cdse")
    ap.add_argument("--aoi", required=True, help="lon_min,lat_min,lon_max,lat_max")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", default="data/raw/sentinel1")
    ap.add_argument("--max-results", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true", help="只列出不下载")
    args = ap.parse_args()

    lon1, lat1, lon2, lat2 = [float(v) for v in args.aoi.split(",")]
    wkt = (f"POLYGON(({lon1} {lat1},{lon2} {lat1},{lon2} {lat2},"
           f"{lon1} {lat2},{lon1} {lat1}))")

    if args.source == "cdse":
        items = _cdse_search(wkt, args.start, args.end, args.max_results)
        logger.info("检索到 %d 景 IW GRDH 产品", len(items))
        for it in items:
            logger.info("  %s | %s | %.2f GB", it["name"], it["start"],
                        (it["size"] or 0) / 1e9)
        if args.dry_run or not items:
            return
        _cdse_download(items, Path(args.out), _cdse_token())
        return

    try:
        import asf_search as asf
    except ImportError:
        logger.error("缺少 asf_search：pip install asf_search")
        sys.exit(1)

    results = asf.search(
        platform=asf.PLATFORM.SENTINEL1,
        processingLevel=asf.PRODUCT_TYPE.GRD_HD,  # IW GRD 高分辨率
        beamMode=asf.BEAMMODE.IW,
        intersectsWith=wkt,
        start=args.start,
        end=args.end,
        maxResults=args.max_results,
    )
    logger.info("检索到 %d 景 IW GRD 产品", len(results))
    for r in results:
        props = r.properties
        logger.info("  %s | %s | %s", props.get("sceneName"),
                    props.get("startTime"), props.get("polarization"))

    if args.dry_run:
        return

    user, pwd = os.environ.get("EARTHDATA_USER"), os.environ.get("EARTHDATA_PASS")
    if not user or not pwd:
        logger.error("未设置 EARTHDATA_USER / EARTHDATA_PASS 环境变量（需先注册账号）")
        sys.exit(1)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    session = asf.ASFSession().auth_with_creds(user, pwd)
    results.download(path=str(out), session=session, processes=4)
    logger.info("下载完成 → %s", out)


if __name__ == "__main__":
    main()
