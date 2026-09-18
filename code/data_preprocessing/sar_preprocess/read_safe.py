"""纯 Python 读取 Sentinel-1 IW GRD SAFE 产品（不依赖 SNAP 的降级链路）。

输入可以是 .SAFE 目录，也可以是 zip 流（含 .SAFE 后缀的 zip 文件，
与 SNAP 下载的原始压缩包结构一致）。zip 情况只解压需要的成员
（measurement 的 GeoTIFF + annotation/calibration 的 XML）到临时目录，
处理完自动清理；也可用 extract_dir 指定保留的解压位置。

定标：σ0 = DN² / A²，A 为 calibration XML 中 sigmaNought 的
calibrationVector，行内按 pixel 线性插值、行间按 line 线性插值。
入射角：annotation XML 的 geolocationGridPoint 双线性插值到全分辨率。
配准：用 GCP 网格四角拟合 EPSG:4326 下的近似仿射变换
（GRD 像素在地面距离向等间隔，经纬度下只是近似，足够给出经纬度范围）。

输出 dict 与 tile.read_bands 一致：{"vv","vh","incidence","crs","transform"}，
vv/vh 为 σ0 线性值 float32，incidence 单位为度。
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
from utils.logger import get_logger

logger = get_logger("read_safe")


def _local(tag: str) -> str:
    """去掉 XML 命名空间，取本地标签名。"""
    return tag.rsplit("}", 1)[-1]


def _find_one(root: Path, pattern: str) -> Path:
    hits = sorted(root.rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"{root} 下找不到 {pattern}")
    return hits[0]


def _prepare_safe_dir(path: Path, extract_dir: Path | None):
    """统一入口：返回 (safe_dir, 临时目录或 None)。

    目录直接用；zip 流（.zip 或 .SAFE 文件）只解出 measurement 与
    annotation 成员，避免无谓地解压 manifest/preview 等。
    """
    if path.is_dir():
        return path, None
    if not zipfile.is_zipfile(path):
        raise ValueError(f"{path} 既不是目录也不是 zip 流")
    tmp = Path(extract_dir) if extract_dir else Path(
        tempfile.mkdtemp(prefix="safe_extract_"))
    tmp.mkdir(parents=True, exist_ok=True)
    logger.info("解压 zip 流（仅 measurement/annotation）→ %s", tmp)
    with zipfile.ZipFile(path) as zf:
        members = [n for n in zf.namelist()
                   if "/measurement/" in n or "/annotation/" in n]
        if not members:
            raise FileNotFoundError(f"{path} 内没有 SAFE 结构成员")
        top = members[0].split("/")[0]  # 顶层 <场景名>.SAFE/
        for m in members:
            zf.extract(m, tmp)
    return tmp / top, (None if extract_dir else tmp)


def _parse_calibration(cal_xml: Path):
    """解析 calibration XML → (lines, [各行(pixel采样点, sigmaNought)]）。

    calibrationVector 的 line 是该定标行对应的影像行号（含），
    pixel/sigmaNought 是等间隔采样点序列。
    """
    tree = ET.parse(cal_xml)
    lines, vectors = [], []
    for cv in tree.iter():
        if _local(cv.tag) != "calibrationVector":
            continue
        line = pix = sig = None
        for child in cv:
            name = _local(child.tag)
            if name == "line":
                line = int(child.text)
            elif name == "pixel":
                pix = np.fromstring(child.text, sep=" ")
            elif name == "sigmaNought":
                sig = np.fromstring(child.text, sep=" ", dtype=np.float64)
        if line is not None and pix is not None and sig is not None:
            lines.append(line)
            vectors.append((pix, sig))
    if not lines:
        raise ValueError(f"{cal_xml} 中没有 calibrationVector")
    order = np.argsort(lines)
    return ([lines[i] for i in order],
            [vectors[i] for i in order])


def _parse_geolocation(ann_xml: Path):
    """解析 annotation XML 的 geolocationGridPoint → (grid_lines,
    grid_pixels, {incidence/latitude/longitude: [n_line, n_pixel]}）。"""
    tree = ET.parse(ann_xml)
    recs = []
    for gp in tree.iter():
        if _local(gp.tag) != "geolocationGridPoint":
            continue
        rec = {}
        for child in gp:
            name = _local(child.tag)
            if name in ("line", "pixel"):
                rec[name] = int(child.text)
            elif name in ("latitude", "longitude", "incidenceAngle"):
                rec[name] = float(child.text)
        if rec:
            recs.append(rec)
    if not recs:
        raise ValueError(f"{ann_xml} 中没有 geolocationGridPoint")

    grid_lines = np.array(sorted({r["line"] for r in recs}))
    grid_pixels = np.array(sorted({r["pixel"] for r in recs}))
    li = {v: i for i, v in enumerate(grid_lines)}
    pi = {v: i for i, v in enumerate(grid_pixels)}
    grids = {}
    for key in ("incidenceAngle", "latitude", "longitude"):
        g = np.full((len(grid_lines), len(grid_pixels)), np.nan)
        for r in recs:
            g[li[r["line"]], pi[r["pixel"]]] = r[key]
        grids[key.split("Angle")[0] if key == "incidenceAngle" else key] = g
    return grid_lines, grid_pixels, grids


def _interp_grid(grid_lines: np.ndarray, grid_pixels: np.ndarray,
                 values: np.ndarray, height: int, width: int,
                 chunk_rows: int = 2048) -> np.ndarray:
    """规则 GCP 网格双线性插值到全分辨率（分块控制内存）。

    先沿 pixel 方向对每个网格行插值到全宽，再沿 line 方向逐行线性混合。
    """
    cols = np.arange(width)
    # 第一阶段：[n_grid_line, W]
    wide = np.stack([np.interp(cols, grid_pixels, row)
                     for row in values]).astype(np.float32)
    rows = np.arange(height, dtype=np.float64)
    # 每个输出行落在哪两条网格线之间（越界取端点）
    idx = np.clip(np.searchsorted(grid_lines, rows) - 1,
                  0, len(grid_lines) - 2)
    span = grid_lines[idx + 1] - grid_lines[idx]
    w = np.where(span > 0, (rows - grid_lines[idx]) / np.maximum(span, 1),
                 0.0)
    w = np.clip(w, 0.0, 1.0).astype(np.float32)  # 越界行取端点，不外推
    out = np.empty((height, width), dtype=np.float32)
    for y0 in range(0, height, chunk_rows):
        y1 = min(y0 + chunk_rows, height)
        sl = slice(y0, y1)
        out[sl] = ((1 - w[sl, None]) * wide[idx[sl]]
                   + w[sl, None] * wide[idx[sl] + 1])
    return out


def _calibrate(tif_path: Path, cal_xml: Path) -> np.ndarray:
    """σ0 = DN² / A²，按定标行分段处理（行间线性插值 A）。"""
    import rasterio

    lines, vectors = _parse_calibration(cal_xml)
    with rasterio.open(tif_path) as src:
        height, width = src.height, src.width
        # 每条定标行先内插到全宽
        full_vecs = [np.interp(np.arange(width), pix, sig)
                     for pix, sig in vectors]
        sigma0 = np.empty((height, width), dtype=np.float32)
        # 分段：[首行之前]、[line_i, line_{i+1})、[末行之后]
        bounds = [0] + [l for l in lines if 0 < l < height] + [height]
        for y0, y1 in zip(bounds[:-1], bounds[1:]):
            dn = src.read(1, window=((y0, y1), (0, width))).astype(np.float32)
            i = min(max(np.searchsorted(lines, y0, side="right") - 1, 0),
                    len(lines) - 2)
            l0, l1 = lines[i], lines[min(i + 1, len(lines) - 1)]
            a0 = full_vecs[i]
            a1 = full_vecs[min(i + 1, len(lines) - 1)]
            rows = np.arange(y0, y1, dtype=np.float32)
            w = 0.0 if l1 == l0 else np.clip((rows - l0) / (l1 - l0), 0, 1)
            a = (1 - w[:, None]) * a0 + w[:, None] * a1
            sigma0[y0:y1] = dn * dn / np.maximum(a * a, 1e-20)
        # DN=0 是刈幅外/缺失像元，置 NaN（否则会污染 dB 统计与归一化）
        sigma0[sigma0 == 0] = np.nan
    return sigma0


def _affine_from_gcp(grid_lines: np.ndarray, grid_pixels: np.ndarray,
                     grids: dict) -> tuple:
    """用 GCP 网格拟合 EPSG:4326 近似仿射变换 (a,b,c,d,e,f)。

    分辨率 = 网格两端经纬度差 / 两端 line/pixel 坐标差，再外推网格
    首点前与末点后的部分（含半个像元边缘）。GRD 是地面距离等间隔，
    经纬度下只是近似配准，用于给出经纬度范围与块级定位。
    """
    lat, lon = grids["latitude"], grids["longitude"]
    # 端点取网格四端（首行首尾、末行首尾）平均，避免单边倾斜带来偏差
    lat_top = float(np.mean(lat[0, [0, -1]]))
    lat_bot = float(np.mean(lat[-1, [0, -1]]))
    lon_left = float(np.mean(lon[[0, -1], 0]))
    lon_right = float(np.mean(lon[[0, -1], -1]))
    dlon = (lon_right - lon_left) / (grid_pixels[-1] - grid_pixels[0])
    dlat = (lat_bot - lat_top) / (grid_lines[-1] - grid_lines[0])
    # 网格点是像素中心：原点 = 首网格点经纬度 - 其像素坐标×分辨率 - 半像元
    return (dlon, 0.0, lon_left - grid_pixels[0] * dlon - dlon / 2,
            0.0, dlat, lat_top - grid_lines[0] * dlat - dlat / 2)


def read_safe(path, extract_dir=None) -> dict:
    """读 SAFE 产品 → {"vv","vh","incidence","crs","transform"}。

    path: .SAFE 目录，或 zip 流（.zip / .SAFE 文件）。
    extract_dir: zip 解压位置（None 则用临时目录、用完删除）。
    """
    path = Path(path)
    safe_dir, tmp = _prepare_safe_dir(path, extract_dir)
    try:
        logger.info("读取 SAFE 产品：%s", safe_dir.name)
        tif_vv = _find_one(safe_dir, "measurement/*-vv-*.tiff")
        tif_vh = _find_one(safe_dir, "measurement/*-vh-*.tiff")
        cal_vv = _find_one(safe_dir,
                           "annotation/calibration/calibration-*-vv-*.xml")
        cal_vh = _find_one(safe_dir,
                           "annotation/calibration/calibration-*-vh-*.xml")
        ann_vv = _find_one(safe_dir, "annotation/*-vv-*.xml")

        logger.info("定标 VV …")
        vv = _calibrate(tif_vv, cal_vv)
        logger.info("定标 VH …")
        vh = _calibrate(tif_vh, cal_vh)

        grid_lines, grid_pixels, grids = _parse_geolocation(ann_vv)
        h, w = vv.shape
        logger.info("入射角网格 %d×%d 插值到 %d×%d …",
                    len(grid_lines), len(grid_pixels), h, w)
        inc = _interp_grid(grid_lines, grid_pixels,
                           grids["incidence"].astype(np.float64), h, w)
        transform = _affine_from_gcp(grid_lines, grid_pixels, grids)
        logger.info("配准：经度 %.4f~%.4f，纬度 %.4f~%.4f",
                    transform[2], transform[2] + w * transform[0],
                    transform[5] + h * transform[4], transform[5])
        return {"vv": vv, "vh": vh, "incidence": inc,
                "crs": "EPSG:4326", "transform": transform}
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
