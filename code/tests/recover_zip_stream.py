"""从截断的 Sentinel-1 zip 流中抢救完整成员。

背景：下载中断的 zip 流没有中央目录，zipfile 打不开；但 local file
header 都在，成员按顺序存放（deflate + data descriptor）。本脚本按
PK\x03\x04 签名扫描定位每个成员，用 zlib 原始 deflate 解压到流自然
结束，能完整解压的成员写出，截断成员尽力解压（尾部数据缺失）。

用法（在 code/ 目录下）：
    python tests/recover_zip_stream.py <zip流> <输出目录>
"""
from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path


def main():
    src = Path(sys.argv[1])
    out_root = Path(sys.argv[2])
    buf = src.read_bytes()
    size = len(buf)

    # 扫描所有 local file header 签名
    offs = []
    i = 0
    while True:
        i = buf.find(b"PK\x03\x04", i)
        if i < 0:
            break
        offs.append(i)
        i += 4

    for k, off in enumerate(offs):
        hdr = buf[off:off + 30]
        method = struct.unpack("<H", hdr[8:10])[0]
        nlen, elen = struct.unpack("<HH", hdr[26:30])
        name = buf[off + 30:off + 30 + nlen].decode("utf-8", "replace")
        data_off = off + 30 + nlen + elen
        end = offs[k + 1] if k + 1 < len(offs) else size
        payload = buf[data_off:end]

        # 流式 zip 的 data descriptor：PK\x07\x08 + crc + csize + usize
        desc = None
        if len(payload) >= 16:
            sig_pos = len(payload) - 16
            if payload[sig_pos:sig_pos + 4] == b"PK\x07\x08":
                _, crc, csize, usize = struct.unpack(
                    "<IIII", payload[sig_pos:sig_pos + 16])
                desc = (csize, usize)
                payload = payload[:sig_pos]

        if method == 0:          # stored：数据原样
            data = payload
        else:                    # deflate：解压到流结束即完整
            d = zlib.decompressobj(-15)
            try:
                data = d.decompress(payload)
                data += d.flush()
            except zlib.error:
                data = b""

        # 完整性判定：有 descriptor 且解压大小与 usize 一致
        if desc and method == 0:
            complete = len(data) == desc[0]
            data = data[:desc[0]]
        elif desc:
            complete = len(data) == desc[1]
        else:                    # 无 descriptor（最后一个成员被截断）
            complete = False

        rel = Path(*name.split("/")[1:])  # 去掉顶层 <场景>.SAFE/
        if not rel.parts:        # 目录条目
            continue
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        print(f"{'完整' if complete else '截断/未知'} {rel} → {len(data)} 字节")


if __name__ == "__main__":
    main()
