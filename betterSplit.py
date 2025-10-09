"""Script utilitário para gerar listas de divisão (train/val/test) a partir
de um conjunto de imagens, máscaras de segmentação e máscaras de área.

O script percorre diretórios, casa arquivos que compartilham o mesmo nome
base (ignora diferenças de maiúsculas/minúsculas) e grava arquivos de texto
com as combinações resultantes.
"""

import os
import argparse
import random
import re
from typing import List, Dict, Tuple

IMG_EXTS_DEFAULT = [".jpg", ".jpeg", ".png"]

def list_images(img_dir: str, img_exts: List[str]) -> Dict[str, str]:
    """Percorre ``img_dir`` e monta um dicionário nome_base -> caminho.

    Parameters
    ----------
    img_dir : str
        Pasta raiz com as imagens originais.
    img_exts : List[str]
        Conjunto de extensões aceitas (em minúsculas, incluindo ponto).

    Returns
    -------
    Dict[str, str]
        Mapa com o nome base em lowercase como chave e o caminho absoluto
        do arquivo como valor. Caso haja colisão (nomes iguais em pastas
        diferentes), um aviso é impresso.
    """

    # Dicionário a ser preenchido com a correspondência entre nome base e caminho
    m = {}

    # Caminha recursivamente pelos diretórios e filtra apenas arquivos válidos
    for root, _, files in os.walk(img_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in img_exts:
                # Obtém o nome base (sem extensão) e normaliza para case-insensitive
                base = os.path.splitext(f)[0]
                key = base.lower()  # chave case-insensitive
                path = os.path.join(root, f)

                # Se outro arquivo com o mesmo nome base já existe, alerta o usuário
                if key in m and os.path.normcase(m[key]) != os.path.normcase(path):
                    print(f"AVISO: colisão por nome (case-insensitive): {m[key]} vs {path}")
                m[key] = path
    return m

def list_labels(lbl_dir: str, lbl_ext: str, lbl_suffix: str) -> Dict[str, str]:
    """Indexa arquivos de máscara de classe (labels) removendo um sufixo opcional."""

    m = {}
    # Compila regex para remover o sufixo (caso exista) de forma case-insensitive
    suf_re = re.compile(rf'({re.escape(lbl_suffix)})$', flags=re.IGNORECASE) if lbl_suffix else None
    for root, _, files in os.walk(lbl_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() == lbl_ext.lower():
                base = os.path.splitext(f)[0]
                if suf_re:
                    # Remove o sufixo do nome base para viabilizar a comparação
                    base = suf_re.sub('', base)
                key = base.lower()  # chave case-insensitive
                path = os.path.join(root, f)
                if key in m and os.path.normcase(m[key]) != os.path.normcase(path):
                    print(f"AVISO: colisão por nome (case-insensitive): {m[key]} vs {path}")
                m[key] = path
    return m

def list_masks(mask_dir: str, mask_ext: str, mask_suffix: str) -> Dict[str, str]:
    """Indexa arquivos de máscara de área removendo um sufixo opcional."""

    m = {}
    suf_re = re.compile(rf'({re.escape(mask_suffix)})$', flags=re.IGNORECASE) if mask_suffix else None
    for root, _, files in os.walk(mask_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() == mask_ext.lower():
                base = os.path.splitext(f)[0]
                if suf_re:
                    # Remove o sufixo configurado para permitir casar pelo nome base
                    base = suf_re.sub('', base)
                key = base.lower()  # chave case-insensitive
                path = os.path.join(root, f)
                if key in m and os.path.normcase(m[key]) != os.path.normcase(path):
                    print(f"AVISO: colisão por nome (case-insensitive): {m[key]} vs {path}")
                m[key] = path
    return m

def make_pairs(
    imgs: Dict[str, str],
    lbls: Dict[str, str],
    masks: Dict[str, str]
) -> Tuple[List[Tuple[str, str, str]], List[str], List[str], List[str]]:
    """Intersecta imagens, labels e máscaras de área, reportando ausências."""

    # Calcula os nomes base presentes em todos os conjuntos (imagem, label e máscara)
    common = sorted(set(imgs.keys()) & set(lbls.keys()) & set(masks.keys()))
    # Constrói uma lista com tuplas (imagem, label, máscara)
    pairs = [(imgs[b], lbls[b], masks[b]) for b in common]
    # Identifica possíveis ausências para reportar ao usuário
    missing_lbl = sorted(set(imgs.keys()) - set(lbls.keys()))
    missing_img = sorted(set(lbls.keys()) - set(imgs.keys()))
    missing_mask = sorted(set(imgs.keys()) - set(masks.keys()))
    return pairs, missing_lbl, missing_img, missing_mask

def write_list(out_path: str, pairs: List[Tuple[str, str, str]], out_root: str):
    """Escreve cada trio (imagem, label, máscara) no arquivo solicitado."""

    with open(out_path, "w", encoding="utf-8") as f:
        for ip, lp, mp in pairs:
            # Relativiza os caminhos em relação a ``out_root`` e padroniza separador
            img_rel = os.path.relpath(ip, out_root).replace("\\", "/")
            lbl_rel = os.path.relpath(lp, out_root).replace("\\", "/")
            mask_rel = os.path.relpath(mp, out_root).replace("\\", "/")
            f.write(f"{img_rel} {lbl_rel} {mask_rel}\n")

def main():
    """Configura a linha de comando, prepara os dicionários e gera os splits."""

    ap = argparse.ArgumentParser(description="Cria train/val/test.txt casando imagens, labels e máscaras de área por nome base.")
    # Argumentos obrigatórios: diretórios de imagens e labels
    ap.add_argument("img_dir", help="Diretório de imagens (ex: data/dimensionamento_foliar/images)")
    ap.add_argument("lbl_dir", help="Diretório de labels (ex: data/dimensionamento_foliar/labels)")
    # Argumento opcional para o diretório de máscaras de área
    ap.add_argument("--area-mask-dir", default="data/dimensionamento_foliar/area_masks", help="Diretório de labels de área (opcional, ex: data/dimensionamento_foliar/area_labels)")
    ap.add_argument("--area-mask-ext", type=str, default=".raw", help="Extensão das máscaras de área (ex: .raw)")
    ap.add_argument("--area-mask-suffix", type=str, default="", help="Sufixo removido do nome base da máscara de área")
    # Onde os arquivos finais serão gravados
    ap.add_argument("--out-root", default=None, help="Raiz onde salvar train.txt/val.txt/test.txt (default: raiz comum)")
    ap.add_argument("--splits", type=float, nargs=3, default=[0.8, 0.2, 0.0], metavar=("TRAIN","VAL","TEST"))
    ap.add_argument("--img-exts", type=str, default=",".join(IMG_EXTS_DEFAULT), help="Extensões de imagem, separadas por vírgula")
    ap.add_argument("--lbl-ext", type=str, default=".png", help="Extensão das labels (ex: .png)")
    ap.add_argument("--lbl-suffix", type=str, default="_label", help="Sufixo removido do nome base da label")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    # Normaliza caminhos absolutos para evitar problemas com arquivos relativos
    img_dir = os.path.abspath(args.img_dir)
    lbl_dir = os.path.abspath(args.lbl_dir)
    area_mask_dir = os.path.abspath(args.area_mask_dir)
    out_root = os.path.abspath(args.out_root) if args.out_root else os.path.commonpath([img_dir, lbl_dir])
    # Garante que as extensões tenham o ponto inicial e estejam em minúsculas
    img_exts = [e.strip().lower() if e.strip().startswith(".") else "."+e.strip().lower() for e in args.img_exts.split(",")]
    lbl_ext = args.lbl_ext if args.lbl_ext.startswith(".") else "."+args.lbl_ext
    mask_ext = args.area_mask_ext if args.area_mask_ext.startswith(".") else "." + args.area_mask_ext

    # Cria os mapas de nome base -> caminho para imagens, labels e máscaras
    imgs = list_images(img_dir, img_exts)
    lbls = list_labels(lbl_dir, lbl_ext, args.lbl_suffix)
    masks = list_masks(area_mask_dir, mask_ext, args.area_mask_suffix)
    pairs, miss_lbl, miss_img, miss_mask = make_pairs(imgs, lbls, masks)

    print("Pareamento case-insensitive por nome base (ignorando maiúsculas/minúsculas).")
    print(f"Imagens: {len(imgs)} | Labels: {len(lbls)} | Máscaras: {len(masks)} | Pares: {len(pairs)}")

    # Reporta possíveis inconsistências encontradas durante o pareamento
    if miss_lbl:
        print(f"Imagens sem label: {len(miss_lbl)} (ex.: {miss_lbl[:5]})")
    if miss_img:
        print(f"Labels sem imagem: {len(miss_img)} (ex.: {miss_img[:5]})")
    if miss_mask:
        print(f"Imagens sem máscara de área: {len(miss_mask)} (ex.: {miss_mask[:5]})")

    # Recupera as proporções desejadas e verifica se somam 1.0
    s_train, s_val, s_test = args.splits
    total = len(pairs)
    if abs(s_train + s_val + s_test - 1.0) > 1e-6:
        raise ValueError("Splits devem somar 1.0")
    # Embaralha os pares para garantir aleatoriedade reprodutível
    random.seed(args.seed)
    random.shuffle(pairs)
    # Converte as proporções em contagens absolutas
    n_train = int(total * s_train)
    n_val = int(total * s_val)
    n_test = total - n_train - n_val

    # Faz o fatiamento dos pares conforme as quantidades calculadas
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train+n_val]
    test_pairs = pairs[n_train+n_val:]

    # Persistência das listas de caminhos relativos
    write_list(os.path.join(out_root, "train.txt"), train_pairs, out_root)
    write_list(os.path.join(out_root, "val.txt"), val_pairs, out_root)
    write_list(os.path.join(out_root, "test.txt"), test_pairs, out_root)

    print(f"train: {len(train_pairs)} | val: {len(val_pairs)} | test: {len(test_pairs)}")
    print(f"Listas salvas em: {out_root}")

if __name__ == "__main__":
    main()