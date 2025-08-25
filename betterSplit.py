import os
import argparse
import random
import re
from typing import List, Dict, Tuple

IMG_EXTS_DEFAULT = [".jpg", ".jpeg", ".png"]

def list_images(img_dir: str, img_exts: List[str]) -> Dict[str, str]:
    m = {}
    for root, _, files in os.walk(img_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in img_exts:
                base = os.path.splitext(f)[0]
                key = base.lower()  # chave case-insensitive
                path = os.path.join(root, f)
                if key in m and os.path.normcase(m[key]) != os.path.normcase(path):
                    print(f"AVISO: colisão por nome (case-insensitive): {m[key]} vs {path}")
                m[key] = path
    return m

def list_labels(lbl_dir: str, lbl_ext: str, lbl_suffix: str) -> Dict[str, str]:
    m = {}
    suf_re = re.compile(rf'({re.escape(lbl_suffix)})$', flags=re.IGNORECASE) if lbl_suffix else None
    for root, _, files in os.walk(lbl_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() == lbl_ext.lower():
                base = os.path.splitext(f)[0]
                if suf_re:
                    base = suf_re.sub('', base)
                key = base.lower()  # chave case-insensitive
                path = os.path.join(root, f)
                if key in m and os.path.normcase(m[key]) != os.path.normcase(path):
                    print(f"AVISO: colisão por nome (case-insensitive): {m[key]} vs {path}")
                m[key] = path
    return m

def make_pairs(imgs: Dict[str, str], lbls: Dict[str, str]) -> Tuple[List[Tuple[str,str]], List[str], List[str]]:
    common = sorted(set(imgs.keys()) & set(lbls.keys()))
    pairs = [(imgs[b], lbls[b]) for b in common]
    missing_lbl = sorted(set(imgs.keys()) - set(lbls.keys()))
    missing_img = sorted(set(lbls.keys()) - set(imgs.keys()))
    return pairs, missing_lbl, missing_img

def write_list(out_path: str, pairs: List[Tuple[str,str]], out_root: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for ip, lp in pairs:
            img_rel = os.path.relpath(ip, out_root).replace("\\", "/")
            lbl_rel = os.path.relpath(lp, out_root).replace("\\", "/")
            f.write(f"{img_rel} {lbl_rel}\n")

def main():
    ap = argparse.ArgumentParser(description="Cria train/val/test.txt casando imagens e labels por nome base.")
    ap.add_argument("img_dir", help="Diretório de imagens (ex: data/dimensionamento_foliar/images)")
    ap.add_argument("lbl_dir", help="Diretório de labels (ex: data/dimensionamento_foliar/labels)")
    ap.add_argument("--out-root", default=None, help="Raiz onde salvar train.txt/val.txt/test.txt (default: raiz comum)")
    ap.add_argument("--splits", type=float, nargs=3, default=[0.8, 0.2, 0.0], metavar=("TRAIN","VAL","TEST"))
    ap.add_argument("--img-exts", type=str, default=",".join(IMG_EXTS_DEFAULT), help="Extensões de imagem, separadas por vírgula")
    ap.add_argument("--lbl-ext", type=str, default=".png", help="Extensão das labels (ex: .png)")
    ap.add_argument("--lbl-suffix", type=str, default="_label", help="Sufixo removido do nome base da label")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    img_dir = os.path.abspath(args.img_dir)
    lbl_dir = os.path.abspath(args.lbl_dir)
    out_root = os.path.abspath(args.out_root) if args.out_root else os.path.commonpath([img_dir, lbl_dir])
    img_exts = [e.strip().lower() if e.strip().startswith(".") else "."+e.strip().lower() for e in args.img_exts.split(",")]
    lbl_ext = args.lbl_ext if args.lbl_ext.startswith(".") else "."+args.lbl_ext

    imgs = list_images(img_dir, img_exts)
    lbls = list_labels(lbl_dir, lbl_ext, args.lbl_suffix)
    pairs, miss_lbl, miss_img = make_pairs(imgs, lbls)

    print("Pareamento case-insensitive por nome base (ignorando maiúsculas/minúsculas).")
    print(f"Imagens: {len(imgs)} | Labels: {len(lbls)} | Pares: {len(pairs)}")

    if miss_lbl:
        print(f"Imagens sem label: {len(miss_lbl)} (ex.: {miss_lbl[:5]})")
    if miss_img:
        print(f"Labels sem imagem: {len(miss_img)} (ex.: {miss_img[:5]})")

    s_train, s_val, s_test = args.splits
    total = len(pairs)
    if abs(s_train + s_val + s_test - 1.0) > 1e-6:
        raise ValueError("Splits devem somar 1.0")
    random.seed(args.seed)
    random.shuffle(pairs)
    n_train = int(total * s_train)
    n_val = int(total * s_val)
    n_test = total - n_train - n_val

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train+n_val]
    test_pairs = pairs[n_train+n_val:]

    write_list(os.path.join(out_root, "train.txt"), train_pairs, out_root)
    write_list(os.path.join(out_root, "val.txt"), val_pairs, out_root)
    write_list(os.path.join(out_root, "test.txt"), test_pairs, out_root)

    print(f"train: {len(train_pairs)} | val: {len(val_pairs)} | test: {len(test_pairs)}")
    print(f"Listas salvas em: {out_root}")

if __name__ == "__main__":
    main()