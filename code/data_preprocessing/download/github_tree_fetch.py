"""按 GitHub tree 清单逐文件下载仓库内容（raw.githubusercontent.com 直连）。

背景：repo zip 经代理下载反复超时失败，且代理不支持断点；
raw 直连可用，逐文件下载天然支持断点（已存在且大小一致的跳过）。

用法：
    python data_preprocessing/download/github_tree_fetch.py \
        --tree data/raw/air_kaggle_iw/gh_tree.json \
        --repo Sushmit1/Automated-Detection-of-Oceanic-Internal-Waves \
        --out data/raw/air_kaggle_iw/repo --prefix dataset/ --workers 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))
from utils.logger import get_logger

logger = get_logger("github_tree_fetch")

RAW = "https://raw.githubusercontent.com/{repo}/main/{path}"


def fetch_one(repo: str, path: str, dest: Path, size: int,
              retries: int = 5, proxy: str = "") -> str:
    import requests

    if dest.exists() and dest.stat().st_size == size:
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = proxy + RAW.format(repo=repo, path=path)
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200 and len(r.content) == size:
                dest.write_bytes(r.content)
                return "ok"
            logger.warning("%s 状态 %s 大小 %s（期望 %d），重试 %d/%d",
                           path, r.status_code, len(r.content), size,
                           attempt + 1, retries)
        except requests.RequestException as e:
            logger.warning("%s 异常 %s，重试 %d/%d", path, e, attempt + 1, retries)
        time.sleep(2 * (attempt + 1))
    return "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True, help="GitHub git/trees API 返回的 json")
    ap.add_argument("--repo", required=True, help="owner/repo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="dataset/", help="只下该前缀路径")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--proxy", default="",
                    help="URL 前缀代理，如 https://gh-proxy.com/")
    args = ap.parse_args()

    tree = json.loads(Path(args.tree).read_text(encoding="utf-8"))["tree"]
    files = [t for t in tree if t["type"] == "blob"
             and t["path"].startswith(args.prefix)]
    total_mb = sum(t.get("size", 0) for t in files) / 1e6
    logger.info("待下载 %d 个文件，共 %.1f MB", len(files), total_mb)

    out = Path(args.out)
    counts = {"ok": 0, "skip": 0, "fail": 0}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, args.repo, t["path"],
                          out / t["path"], t.get("size", -1),
                          proxy=args.proxy): t["path"]
                for t in files}
        for i, fut in enumerate(as_completed(futs), 1):
            counts[fut.result()] += 1
            if i % 500 == 0:
                logger.info("进度 %d/%d（ok %d skip %d fail %d）耗时 %.0f s",
                            i, len(futs), counts["ok"], counts["skip"],
                            counts["fail"], time.time() - t0)
    logger.info("完成：ok %d skip %d fail %d，耗时 %.0f s",
                counts["ok"], counts["skip"], counts["fail"], time.time() - t0)
    if counts["fail"]:
        logger.error("有 %d 个文件失败，重跑本命令即可续传补齐", counts["fail"])
        sys.exit(2)


if __name__ == "__main__":
    main()
