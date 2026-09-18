"""read_safe 单元级验证：合成迷你 SAFE 产品（目录 + zip 流两种输入）。

构造 60×80 的假 GRD 场景：DN 已知、calibrationVector 与
geolocationGridPoint 已知，验证 σ0 定标值、入射角双线性插值、
仿射变换的经纬度范围是否符合手算结果。
用法（在 code/ 目录下）：
    python tests/test_read_safe.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import rasterio
from rasterio.transform import from_origin

from data_preprocessing.sar_preprocess.read_safe import read_safe

H, W = 60, 80
SCENE = "S1A_IW_GRDH_1SDV_20230101T000000_20230101T000025_000000_000000_TEST.SAFE"


def _write_tif(path: Path, dn: np.ndarray):
    with rasterio.open(path, "w", driver="GTiff", height=H, width=W,
                       count=1, dtype="uint16") as dst:
        dst.write(dn, 1)


def _cal_xml(n_lines=3, n_pix=4):
    """sigmaNought = 100 + line + pixel/100（已知函数，便于手算校验）。"""
    vecs = []
    lines = np.linspace(0, H - 1, n_lines).astype(int)
    pixels = np.linspace(0, W - 1, n_pix).astype(int)
    for ln in lines:
        sig = [100 + ln + p / 100 for p in pixels]
        vecs.append(
            f"<calibrationVector><line>{ln}</line>"
            f"<pixel>{' '.join(map(str, pixels))}</pixel>"
            f"<sigmaNought>{' '.join(f'{s:.6e}' for s in sig)}</sigmaNought>"
            f"</calibrationVector>")
    return ("<calibration><calibrationVectorList count="
            f"\"{n_lines}\">{''.join(vecs)}</calibrationVectorList>"
            "</calibration>")


def _ann_xml(n_lines=3, n_pix=4):
    """incidence = 30 + line/10 + pixel/100；lat/lon 线性网格。"""
    lines = np.linspace(0, H - 1, n_lines).astype(int)
    pixels = np.linspace(0, W - 1, n_pix).astype(int)
    gps = []
    for ln in lines:
        for px in pixels:
            lat = 20.0 - ln * 0.01
            lon = 110.0 + px * 0.01
            inc = 30 + ln / 10 + px / 100
            gps.append(
                "<geolocationGridPoint>"
                f"<line>{ln}</line><pixel>{px}</pixel>"
                f"<latitude>{lat}</latitude><longitude>{lon}</longitude>"
                f"<height>0</height><incidenceAngle>{inc}</incidenceAngle>"
                "</geolocationGridPoint>")
    return ("<product><geolocationGrid>"
            f"<geolocationGridPointList count=\"{len(gps)}\">"
            f"{''.join(gps)}</geolocationGridPointList>"
            "</geolocationGrid></product>")


def _build_safe(root: Path) -> Path:
    safe = root / SCENE
    (safe / "measurement").mkdir(parents=True)
    (safe / "annotation" / "calibration").mkdir(parents=True)
    rng = np.random.RandomState(0)
    for pol in ("vv", "vh"):
        dn = rng.randint(50, 500, size=(H, W)).astype(np.uint16)
        _write_tif(safe / "measurement" / f"s1a-iw-grd-{pol}-test.tiff", dn)
        (safe / "annotation" / "calibration"
         / f"calibration-s1a-iw-grd-{pol}-test.xml").write_text(
            _cal_xml(), encoding="utf-8")
        (safe / "annotation"
         / f"s1a-iw-grd-{pol}-test.xml").write_text(
            _ann_xml(), encoding="utf-8")
    # 保存 DN 供校验
    return safe


def _check(bands: dict, safe: Path):
    import rasterio as rio
    dn = rio.open(safe / "measurement" / "s1a-iw-grd-vv-test.tiff").read(1)
    y, x = 30, 40  # 网格正中，便于手算
    a = 100 + y + x / 100  # 行/列方向都是线性，插值应精确还原
    expect = dn[y, x] ** 2 / a ** 2
    assert abs(bands["vv"][y, x] - expect) / expect < 1e-4, \
        f"σ0 不符：{bands['vv'][y, x]} vs {expect}"
    assert bands["vv"].dtype == np.float32
    # 入射角：线性场，双线性插值应精确
    assert abs(bands["incidence"][y, x] - (30 + y / 10 + x / 100)) < 0.05
    assert bands["incidence"].shape == (H, W)
    # 仿射：范围应覆盖 lat 20→19.41, lon 110→110.79 附近
    tf = bands["transform"]
    lon0, lat0 = tf[2], tf[5]
    lon1, lat1 = lon0 + W * tf[0], lat0 + H * tf[4]
    assert abs(lon0 - 110) < 0.01 and abs(lon1 - 110.79) < 0.01
    assert abs(lat0 - 20) < 0.01 and abs(lat1 - 19.41) < 0.01
    assert bands["crs"] == "EPSG:4326"
    print(f"  σ0[{y},{x}]={bands['vv'][y, x]:.6f}（期望 {expect:.6f}）")
    print(f"  入射角[{y},{x}]={bands['incidence'][y, x]:.3f}°（期望 33.400°）")
    print(f"  经纬度范围：lon {lon0:.3f}~{lon1:.3f}, lat {lat1:.3f}~{lat0:.3f}")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="test_read_safe_"))
    try:
        safe = _build_safe(tmp)
        print("[1] 目录输入")
        _check(read_safe(safe), safe)

        print("[2] zip 流输入（.SAFE 后缀的 zip）")
        zip_path = tmp / f"{SCENE}.zip_safe"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in safe.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(tmp))
        _check(read_safe(zip_path), safe)

        print("全部通过")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
