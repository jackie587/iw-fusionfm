"""SWOT L2 KaRIn SSH 产品批量下载（NASA PO.DAAC）。

凭据（不进版本库，见 .gitignore 的 .secrets/）：项目根目录
.secrets/credentials.json（{"earthdata": {"username", "password"}}），
或环境变量 EARTHDATA_USER / EARTHDATA_PASS，或 ~/.netrc。

产品：SWOT_L2_LR_SSH_Expert_2.0（250 m 采样海洋版本）。
注意 calval 红利期（2023-03~07，1 天轨道）与 S1 配对机会高一个量级，优先下载。

用法：
    python data_preprocessing/download/download_swot.py \
        --bbox "109,18,114,23" --start 2023-04-01 --end 2023-06-30 \
        --out data/raw/swot
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))
from utils.logger import get_logger

logger = get_logger("download_swot")

SHORT_NAME = "SWOT_L2_LR_SSH_EXPERT_2.0"


def _earthdata_login():
    import os

    import earthaccess

    user = os.environ.get("EARTHDATA_USER")
    pwd = os.environ.get("EARTHDATA_PASS")
    if not user or not pwd:
        cred_file = PROJECT_ROOT / ".secrets" / "credentials.json"
        if cred_file.exists():
            import json
            creds = json.loads(cred_file.read_text(encoding="utf-8")).get("earthdata")
            if creds:
                user = user or creds["username"]
                pwd = pwd or creds["password"]
    if user and pwd:
        os.environ["EARTHDATA_USERNAME"] = user
        os.environ["EARTHDATA_PASSWORD"] = pwd
        earthaccess.login(strategy="environment")
    else:
        earthaccess.login()  # 退化：读 ~/.netrc
    return earthaccess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", required=True, help="lon_min,lat_min,lon_max,lat_max")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", default="data/raw/swot")
    ap.add_argument("--short-name", default=SHORT_NAME,
                    help="集合名；calval 期（2023-03~07）需 SWOT_L2_LR_SSH_2.0")
    ap.add_argument("--contains", default=None,
                    help="按 native-id 子串过滤（如 _Expert_576_）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    try:
        import earthaccess
    except ImportError:
        logger.error("缺少 earthaccess：pip install earthaccess")
        sys.exit(1)

    lon1, lat1, lon2, lat2 = [float(v) for v in args.bbox.split(",")]
    granules = earthaccess.search_data(
        short_name=args.short_name,
        bounding_box=(lon1, lat1, lon2, lat2),
        temporal=(args.start, args.end),
    )
    if args.contains:
        granules = [g for g in granules
                    if args.contains in g["meta"]["native-id"]]
    logger.info("检索到 %d 个 SWOT granule", len(granules))
    for g in granules:
        logger.info("  %s", g["meta"]["native-id"])

    if args.dry_run:
        return

    earthaccess = _earthdata_login()  # .secrets/credentials.json 或环境变量或 ~/.netrc
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    earthaccess.download(granules, str(out))
    logger.info("下载完成 → %s", out)


if __name__ == "__main__":
    main()
