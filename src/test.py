# inspect_sizes.py
from __future__ import annotations
import os
import argparse
from typing import List, Tuple, Dict
from PIL import Image

IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

def is_image_file(name: str) -> bool:
    n = name.lower()
    ext = os.path.splitext(n)[1]
    return (ext in IMG_EXTS) and not n.startswith("._")

def list_images_sorted(dir_path: str, include_masks: bool) -> List[str]:
    files = []
    for f in os.listdir(dir_path):
        if not is_image_file(f):
            continue
        if (not include_masks) and "_mask" in f.lower():
            continue
        files.append(f)

    # 按数字优先排序（slice_000.tif 排在 slice_001.tif 前）
    import re
    def key_fn(x: str):
        m = re.search(r"(\d+)", x)
        return (int(m.group(1)) if m else -1, x)
    return [os.path.join(dir_path, f) for f in sorted(files, key=key_fn)]

def read_size(p: str) -> Tuple[int, int]:
    with Image.open(p) as im:
        w, h = im.size
    return w, h

def print_case(case_dir: str, include_masks: bool, csv_writer=None) -> None:
    img_paths = list_images_sorted(case_dir, include_masks=include_masks)
    if not img_paths:
        print(f"[WARN] {case_dir} 没有找到图像文件。")
        return

    case_name = os.path.basename(os.path.normpath(case_dir))
    sizes = []
    print(f"\n== {case_name} ==")
    for p in img_paths:
        w, h = read_size(p)
        fname = os.path.basename(p)
        print(f"{fname:>24s}  ->  {w} x {h}")
        sizes.append((fname, w, h))
        if csv_writer:
            csv_writer.write(f"{case_name},{fname},{w},{h}\n")

    # 额外：检查 *_mask 与原图是否同尺寸
    base2size: Dict[str, Tuple[int, int]] = {}
    for fname, w, h in sizes:
        base = fname.lower().replace("_mask", "")
        if base not in base2size:
            base2size[base] = (w, h)
        else:
            # 有同名（原图/掩码），检查尺寸一致
            w0, h0 = base2size[base]
            if (w, h) != (w0, h0):
                print(f"[MISMATCH] {fname} 与对应图尺寸不一致：{w}x{h} vs {w0}x{h0}")

def main():
    ap = argparse.ArgumentParser(description="查看每张图像尺寸（可选导出CSV、检查mask匹配）")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--case_dir", type=str, help="单个 case 目录，例如 ../data/kaggle_3m/case_1")
    group.add_argument("--root_dir", type=str, help="包含多个 case_* 的根目录，例如 ../data/kaggle_3m")
    ap.add_argument("--include_masks", action="store_true", help="包含 *_mask 图像")
    ap.add_argument("--to_csv", type=str, default=None, help="可选，把结果写到CSV文件（列：case,filename,width,height）")
    args = ap.parse_args()

    csv_writer = None
    if args.to_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.to_csv)), exist_ok=True)
        csv_writer = open(args.to_csv, "w", encoding="utf-8")
        csv_writer.write("case,filename,width,height\n")

    try:
        if args.case_dir:
            print_case(args.case_dir, include_masks=args.include_masks, csv_writer=csv_writer)
        else:
            # 遍历 root_dir 下的所有子目录（名字以 case 开头的都算）
            for name in sorted(os.listdir(args.root_dir)):
                sub = os.path.join(args.root_dir, name)
                if os.path.isdir(sub) and name.lower().startswith("case"):
                    print_case(sub, include_masks=args.include_masks, csv_writer=csv_writer)
    finally:
        if csv_writer:
            csv_writer.close()

if __name__ == "__main__":
    main()