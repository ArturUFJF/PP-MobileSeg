import argparse
import csv
import json
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import time

"""
Script de estimacao de metricas fisicas (area, perimetro e comprimento)
a partir de mascaras de segmentacao.

Fluxo geral:
1) Le mascara GT e mascara predita (grayscale de classe ou pseudo-color).
2) Extrai objetos por componente conexo para folha e padrao (quadrado de referencia).
3) Faz casamento GT <-> predicao por IoU de mascara.
4) Opcionalmente le XML para:
    - recuperar medidas reais por folha;
    - recuperar pattern-side;
    - usar fallback do objeto <pattern> quando nao houver classe square nas mascaras.
5) Converte medidas em pixel para unidade fisica usando o quadrado de referencia.
6) Gera CSV final e overlays de depuracao (opcional).
"""


@dataclass
class Component:
    comp_id: int
    mask: np.ndarray
    area_px: float
    perimeter_px: float
    length_px: float
    width_px: float
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]


@dataclass
class XmlLeafInfo:
    leaf_index: int
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    real_area: float
    real_perimeter: float
    real_length: float
    real_width: float


@dataclass
class XmlPatternInfo:
    mask: np.ndarray
    real_area: float
    real_perimeter: float
    real_length: float
    real_width: float
    pattern_side_cm: Optional[float]


def parse_xml_reference_values(xml_path: Path) -> Tuple[Optional[float], Optional[float]]:
    """Lê os valores globais do XML usados no CSV.

    Retorna:
    - capture-distance
    - pattern-side
    """
    if not xml_path.exists():
        return None, None

    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    dist_node = root.find("capture-distance")
    pattern_side_node = root.find("pattern-side")

    dist_cm = None
    if dist_node is not None and dist_node.text is not None:
        try:
            dist_cm = float(dist_node.text)
        except ValueError:
            dist_cm = None

    pattern_side_cm = None
    if pattern_side_node is not None and pattern_side_node.text is not None:
        try:
            pattern_side_cm = float(pattern_side_node.text)
        except ValueError:
            pattern_side_cm = None

    return dist_cm, pattern_side_cm


def parse_args():
    """Define e le argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Estimate leaf metrics from PaddleSeg grayscale masks."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to config JSON.",
    )
    parser.add_argument(
        "--save-overlays",
        action="store_true",
        help="Save debug overlays with matched GT and prediction objects.",
    )
    parser.add_argument(
        "--min-object-px",
        type=int,
        default=20,
        help="Minimum component size (in pixels) to keep.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    """Carrega o arquivo de configuracao JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _decode_color_mask_to_labels(mask_bgr: np.ndarray,
                                 class_colors_rgb: List[List[int]]) -> np.ndarray:
    """
    Decodifica mascara pseudo-color (RGB) para indice de classe por pixel.

    Observacao:
    - OpenCV le imagem em BGR, por isso convertemos para RGB antes de comparar.
    - O indice da cor na lista class_colors_rgb vira o indice da classe.
    """
    # Convert to RGB because OpenCV reads PNG as BGR.
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = mask_rgb.shape
    labels = np.zeros((h, w), dtype=np.uint8)

    # Para cada cor da paleta, marcamos os pixels iguais com o indice da classe.
    for class_idx, rgb in enumerate(class_colors_rgb):
        rgb_arr = np.array(rgb, dtype=np.uint8)
        matches = np.all(mask_rgb == rgb_arr, axis=2)
        labels[matches] = np.uint8(class_idx)
    return labels


def load_mask(path: Path, class_colors_rgb: Optional[List[List[int]]] = None) -> np.ndarray:
    """
    Carrega mascara de segmentacao em formato de indice de classe.

    Suporta dois formatos de entrada:
    - 2D: mascara ja em classe (0,1,2,...)
    - 3D: pseudo-color; converte para classe usando paleta.
    """
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")

    # Standard class-index mask.
    if mask.ndim == 2:
        return mask

    # Color mask (e.g., pseudo_color_prediction).
    if mask.ndim == 3:
        if class_colors_rgb is None:
            # Fallback palette used in this workspace's predict flow.
            class_colors_rgb = [[0, 0, 0], [255, 0, 0], [0, 0, 255]]
        return _decode_color_mask_to_labels(mask, class_colors_rgb)

    raise ValueError(f"Unsupported mask format for {path}: shape={mask.shape}")


def _crop_mask(mask_bool: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Recorta a máscara para a caixa delimitadora fornecida."""
    x, y, w, h = bbox
    return mask_bool[y:y + h, x:x + w]


def estimate_length_width_by_pca(
    mask_bool: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[float, float]:
    """
    Estima comprimento e largura em pixels via PCA da folha.

    Fluxo:
    1) Obtém pontos (x,y) da máscara binária.
    2) Projeta os pontos nos eixos principais (PCA).
    3) Rotaciona a folha para o sistema principal.
    4) Mede a caixa envolvente alinhada aos eixos principais.
    """
    if bbox is not None:
        mask_bool = _crop_mask(mask_bool, bbox)

    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return 0.0, 0.0

    pts = np.column_stack((xs, ys)).astype(np.float32)
    if pts.shape[0] < 3:
        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
        return float(max(w, h)), float(min(w, h))

    mean, eigenvectors = cv2.PCACompute(pts, mean=None)
    if eigenvectors is None or eigenvectors.shape[0] < 2:
        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
        return float(max(w, h)), float(min(w, h))

    proj = cv2.PCAProject(pts, mean, eigenvectors)
    proj_int = np.round(proj).astype(np.int32)
    x, y, w, h = cv2.boundingRect(proj_int)
    length = float(max(w, h))
    width = float(min(w, h))
    return length, width


def connected_components(mask: np.ndarray, class_id: int, min_px: int) -> List[Component]:
    """
    Extrai objetos de uma classe especifica usando componentes conexos.

    Para cada componente, calcula:
    - area em pixel,
    - perimetro,
    - comprimento (aprox. eixo principal via PCA),
    - bbox e centroide.
    """
    # Isola apenas a classe de interesse (ex.: folha=1, padrao=2).
    binary = (mask == class_id).astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    comps: List[Component] = []
    # comp_id=0 e background, por isso iniciamos em 1.
    for comp_id in range(1, n_labels):
        area_stat = float(stats[comp_id, cv2.CC_STAT_AREA])
        if area_stat < min_px:
            continue

        x = int(stats[comp_id, cv2.CC_STAT_LEFT])
        y = int(stats[comp_id, cv2.CC_STAT_TOP])
        w = int(stats[comp_id, cv2.CC_STAT_WIDTH])
        h = int(stats[comp_id, cv2.CC_STAT_HEIGHT])

        comp_mask = labels == comp_id
        comp_crop = _crop_mask(comp_mask, (x, y, w, h))
        comp_u8 = comp_crop.astype(np.uint8)
        # Extrai contornos para calcular perimetro e comprimento.
        contours, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        perimeter = 0.0
        contour_area = 0.0
        for cnt in contours:
            contour_area += float(cv2.contourArea(cnt))
            perimeter += float(cv2.arcLength(cnt, True))

        area = contour_area if contour_area > 0 else area_stat
        length, width = estimate_length_width_by_pca(comp_mask, (x, y, w, h))

        cx, cy = centroids[comp_id]
        comps.append(
            Component(
                comp_id=comp_id,
                mask=comp_mask,
                area_px=area,
                perimeter_px=perimeter,
                length_px=length,
                width_px=width,
                bbox=(x, y, w, h),
                centroid=(float(cx), float(cy)),
            )
        )

    return comps


def connected_components_all(mask: np.ndarray, min_px: int, background_id: int = 0) -> List[Component]:
    """Extrai todos os componentes conexos de uma mascara, ignorando o background."""
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape={mask.shape}")

    foreground = (mask != background_id).astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(foreground, connectivity=8)

    comps: List[Component] = []
    for comp_id in range(1, n_labels):
        area_stat = float(stats[comp_id, cv2.CC_STAT_AREA])
        if area_stat < min_px:
            continue

        x = int(stats[comp_id, cv2.CC_STAT_LEFT])
        y = int(stats[comp_id, cv2.CC_STAT_TOP])
        w = int(stats[comp_id, cv2.CC_STAT_WIDTH])
        h = int(stats[comp_id, cv2.CC_STAT_HEIGHT])

        comp_mask = labels == comp_id
        comp_crop = _crop_mask(comp_mask, (x, y, w, h))
        comp_u8 = comp_crop.astype(np.uint8)
        contours, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        perimeter = 0.0
        contour_area = 0.0
        for cnt in contours:
            contour_area += float(cv2.contourArea(cnt))
            perimeter += float(cv2.arcLength(cnt, True))

        area = contour_area if contour_area > 0 else area_stat
        length, width = estimate_length_width_by_pca(comp_mask, (x, y, w, h))

        cx, cy = centroids[comp_id]
        comps.append(
            Component(
                comp_id=comp_id,
                mask=comp_mask,
                area_px=area,
                perimeter_px=perimeter,
                length_px=length,
                width_px=width,
                bbox=(x, y, w, h),
                centroid=(float(cx), float(cy)),
            )
        )

    return comps


def component_from_binary_mask(mask_bool: np.ndarray) -> Optional[Component]:
    """
    Converte uma mascara binaria unica em um objeto Component.

    Usado principalmente para fallback do pattern vindo do XML.
    """
    comp_u8 = mask_bool.astype(np.uint8)
    if comp_u8.sum() <= 0:
        return None

    # Bounding box por extremos dos pixels ativos.
    ys, xs = np.where(comp_u8 > 0)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    bbox = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
    centroid = (float(xs.mean()), float(ys.mean()))

    # Extrai forma para perimetro/comprimento, igual ao fluxo normal.
    comp_crop = _crop_mask(mask_bool, bbox)
    contours, _ = cv2.findContours(comp_crop.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    perimeter = 0.0
    contour_area = 0.0
    for cnt in contours:
        contour_area += float(cv2.contourArea(cnt))
        perimeter += float(cv2.arcLength(cnt, True))

    area = contour_area if contour_area > 0 else float(comp_u8.sum())
    length, width = estimate_length_width_by_pca(mask_bool, bbox)

    return Component(
        comp_id=1,
        mask=mask_bool,
        area_px=area,
        perimeter_px=perimeter,
        length_px=length,
        width_px=width,
        bbox=bbox,
        centroid=centroid,
    )


def bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Calcula IoU entre duas bounding boxes no formato (x,y,w,h)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = float((ix2 - ix1) * (iy2 - iy1))
    union = float(aw * ah + bw * bh) - inter
    return inter / union if union > 0 else 0.0


def mask_iou(
    a: np.ndarray,
    b: np.ndarray,
    a_bbox: Optional[Tuple[int, int, int, int]] = None,
    b_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> float:
    """Calcula IoU entre duas mascaras binarias, recortando por bbox quando possivel."""
    if a_bbox is not None and b_bbox is not None:
        ax, ay, aw, ah = a_bbox
        bx, by, bw, bh = b_bbox
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        a_slice = a[iy1:iy2, ix1:ix2]
        b_slice = b[iy1:iy2, ix1:ix2]
        inter = float(np.logical_and(a_slice, b_slice).sum())
        union = float(np.logical_or(a_slice, b_slice).sum())
        return inter / union if union > 0 else 0.0

    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def greedy_match(gt_components: List[Component], pred_components: List[Component]) -> List[Tuple[int, int, float, float]]:
    """
    Faz casamento guloso entre componentes GT e preditos por IoU de mascara.

    Retorna lista de tuplas:
    (indice_gt, indice_pred, iou_bbox, iou_mascara)
    """
    candidates: List[Tuple[float, int, int, float]] = []
    # Monta todos os pares validos GT-pred com seus scores.
    for gi, g in enumerate(gt_components):
        for pi, p in enumerate(pred_components):
            iou = mask_iou(g.mask, p.mask, g.bbox, p.bbox)
            if iou <= 0.0:
                continue
            biou = bbox_iou(g.bbox, p.bbox)
            candidates.append((iou, gi, pi, biou))

    # Ordena por IoU de mascara para aplicar matching guloso do melhor para o pior.
    candidates.sort(key=lambda x: x[0], reverse=True)
    used_gt = set()
    used_pred = set()
    matches: List[Tuple[int, int, float, float]] = []
    for iou, gi, pi, biou in candidates:
        # Garante casamento 1-para-1 (cada GT/pred no maximo uma vez).
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append((gi, pi, biou, iou))

    return matches


def match_with_fallback(
    gt_components: List[Component],
    pred_components: List[Component],
) -> Dict[int, Tuple[int, float, float, str]]:
    """
    Faz matching 1-para-1 GT->pred com fallback.

    Ordem:
    1) IoU de máscara;
    2) IoU de bbox entre componentes ainda nao usados;
    3) menor distancia entre centroides entre componentes ainda nao usados.

    Retorna mapa gt_index -> (pred_index, bbox_iou, mask_iou, source).
    """
    matches = greedy_match(gt_components, pred_components)
    used_gt = {gi for gi, _, _, _ in matches}
    used_pred = {pi for _, pi, _, _ in matches}
    out: Dict[int, Tuple[int, float, float, str]] = {
        gi: (pi, biou, miou, "mask") for gi, pi, biou, miou in matches
    }

    for gi, gt_comp in enumerate(gt_components):
        if gi in used_gt:
            continue

        remaining = [
            (pi, pred_comp)
            for pi, pred_comp in enumerate(pred_components)
            if pi not in used_pred
        ]
        if not remaining:
            break

        best_by_bbox = None
        best_bbox_iou = -1.0
        best_bbox_mask_iou = 0.0
        for pi, pred_comp in remaining:
            biou = bbox_iou(gt_comp.bbox, pred_comp.bbox)
            if biou > best_bbox_iou:
                best_bbox_iou = biou
                best_bbox_mask_iou = mask_iou(gt_comp.mask, pred_comp.mask, gt_comp.bbox, pred_comp.bbox)
                best_by_bbox = (pi, pred_comp)

        if best_by_bbox is not None and best_bbox_iou > 0.0:
            pi, _ = best_by_bbox
            used_pred.add(pi)
            out[gi] = (pi, best_bbox_iou, best_bbox_mask_iou, "bbox")
            continue

        # Ultimo fallback: menor distancia entre centroides.
        rx, ry = gt_comp.centroid
        pi, _ = min(
            remaining,
            key=lambda item: (
                (item[1].centroid[0] - rx) ** 2 + (item[1].centroid[1] - ry) ** 2,
            ),
        )
        pred_comp = pred_components[pi]
        used_pred.add(pi)
        out[gi] = (
            pi,
            bbox_iou(gt_comp.bbox, pred_comp.bbox),
            mask_iou(gt_comp.mask, pred_comp.mask, gt_comp.bbox, pred_comp.bbox),
            "centroid",
        )

    return out


def get_xml_points(obj: ET.Element, img_w: int, img_h: int,
                   coord_mode: str) -> List[Tuple[int, int]]:
    """
    Le pontos x1,y1... do XML e converte para pixel.

    coord_mode:
    - "width": x normalizado por largura, y por altura.
    - "height": x e y normalizados por altura (caso legado de alguns XMLs).
    """
    poly = obj.find("polygon")
    base = poly if poly is not None else obj
    points: List[Tuple[int, int]] = []
    idx = 1
    # Le sequencia x1,y1,x2,y2... ate acabar.
    while base.find(f"x{idx}") is not None and base.find(f"y{idx}") is not None:
        x_raw = float(base.find(f"x{idx}").text)
        y_raw = float(base.find(f"y{idx}").text)
        if coord_mode == "height":
            x = int(round(x_raw * img_h))
            y = int(round(y_raw * img_h))
        elif coord_mode == "width":
            x = int(round(x_raw * img_w))
            y = int(round(y_raw * img_h))
        else:
            raise ValueError(f"Unsupported xml coord mode: {coord_mode}")
        points.append((x, y))
        idx += 1
    return points


def polygon_to_mask(points: List[Tuple[int, int]], width: int, height: int) -> np.ndarray:
    """Rasteriza um poligono (lista de pontos) em uma mascara binaria."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(points) >= 3:
        cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], color=1)
    return mask.astype(bool)


def parse_xml_leaf_info(xml_path: Path, width: int, height: int,
                        coord_mode: str) -> Tuple[List[XmlLeafInfo], Optional[float]]:
    """
    Le informacoes de folha no XML:
    - mascara da folha (via poligono),
    - medidas reais (area/perimetro/comprimento),
    - pattern-side (quando presente).
    """
    if not xml_path.exists():
        return [], None

    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    _, pattern_side_cm = parse_xml_reference_values(xml_path)

    leaves: List[XmlLeafInfo] = []
    # Percorre objetos anotados; so folhas com <dimensions> entram no comparativo final.
    for obj in root.findall(".//objects/*"):
        if obj.find("dimensions") is None:
            continue

        points = get_xml_points(obj, width, height, coord_mode)
        if not points:
            continue

        # Le medidas reais da anotacao para calcular RER depois.
        dims = obj.find("dimensions")
        area_txt = dims.find("area").text if dims.find("area") is not None else "-1"
        per_txt = dims.find("perimeter").text if dims.find("perimeter") is not None else "-1"
        len_txt = dims.find("length").text if dims.find("length") is not None else "-1"
        width_txt = dims.find("width").text if dims.find("width") is not None else "-1"

        area = float(area_txt) if area_txt != "NAva" else -1.0
        perimeter = float(per_txt) if per_txt != "NAva" else -1.0
        length = float(len_txt) if len_txt != "NAva" else -1.0
        real_width = float(width_txt) if width_txt != "NAva" else -1.0

        index_node = obj.find("index")
        leaf_idx = -1
        if index_node is not None and index_node.text is not None:
            index_text = index_node.text.strip()
            try:
                leaf_idx = int(index_text)
            except ValueError:
                # Alguns XMLs usam "NAva" ou outros marcadores textuais no campo index.
                # Nesse caso, preservamos o fluxo e marcamos o índice como desconhecido.
                leaf_idx = -1

        leaves.append(
            XmlLeafInfo(
                leaf_index=leaf_idx,
                mask=polygon_to_mask(points, width, height),
                bbox=cv2.boundingRect(np.array(points, dtype=np.int32)),
                real_area=area,
                real_perimeter=perimeter,
                real_length=length,
                real_width=real_width,
            )
        )

    return leaves, pattern_side_cm


def parse_xml_pattern_info(xml_path: Path, width: int, height: int,
                           coord_mode: str) -> Optional[XmlPatternInfo]:
    """
    Le o objeto <pattern> do XML e devolve sua mascara.

    Essa mascara e usada como fallback de referencia fisica quando a
    classe square nao aparece na segmentacao.
    """
    if not xml_path.exists():
        return None

    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    # O objeto <pattern> representa o quadrado de referencia fisica.
    pattern_node = root.find(".//objects/pattern")
    if pattern_node is None:
        return None

    _, pattern_side_cm = parse_xml_reference_values(xml_path)

    points = get_xml_points(pattern_node, width, height, coord_mode)
    if not points:
        return None

    real_area = -1.0
    real_perimeter = -1.0
    real_length = -1.0
    real_width = -1.0
    dims = pattern_node.find("dimensions")
    if dims is not None:
        area_txt = dims.find("area").text if dims.find("area") is not None else "-1"
        per_txt = dims.find("perimeter").text if dims.find("perimeter") is not None else "-1"
        len_txt = dims.find("length").text if dims.find("length") is not None else "-1"
        width_txt = dims.find("width").text if dims.find("width") is not None else "-1"
        real_area = float(area_txt) if area_txt != "NAva" else -1.0
        real_perimeter = float(per_txt) if per_txt != "NAva" else -1.0
        real_length = float(len_txt) if len_txt != "NAva" else -1.0
        real_width = float(width_txt) if width_txt != "NAva" else -1.0
    elif pattern_side_cm is not None:
        real_area = pattern_side_cm * pattern_side_cm
        real_perimeter = pattern_side_cm * 4.0
        real_length = pattern_side_cm
        real_width = pattern_side_cm

    return XmlPatternInfo(
        mask=polygon_to_mask(points, width, height),
        real_area=real_area,
        real_perimeter=real_perimeter,
        real_length=real_length,
        real_width=real_width,
        pattern_side_cm=pattern_side_cm,
    )


def map_gt_to_xml(
    gt_components: List[Component],
    xml_leaves: List[XmlLeafInfo]
) -> Tuple[Dict[int, XmlLeafInfo], float]:
    """
    Casa cada componente GT com uma folha do XML por IoU de mascara.

    Retorna:
    - mapa indice_gt -> XmlLeafInfo
    - score medio de IoU dos casamentos (util para escolher coord_mode).
    """
    if not xml_leaves:
        return {}, 0.0

    # Cria matriz de compatibilidade GT x XML com base em IoU de mascara.
    pairs: List[Tuple[float, int, int]] = []
    for gi, g in enumerate(gt_components):
        for xi, xobj in enumerate(xml_leaves):
            iou = mask_iou(g.mask, xobj.mask, g.bbox, xobj.bbox)
            if iou <= 0:
                continue
            pairs.append((iou, gi, xi))

    # Matching guloso 1-para-1 para evitar multiplos XML no mesmo GT.
    pairs.sort(key=lambda t: t[0], reverse=True)
    used_gt = set()
    used_xml = set()
    out: Dict[int, XmlLeafInfo] = {}
    score_sum = 0.0
    for iou, gi, xi in pairs:
        if gi in used_gt or xi in used_xml:
            continue
        used_gt.add(gi)
        used_xml.add(xi)
        out[gi] = xml_leaves[xi]
        score_sum += iou
    score = score_sum / len(out) if out else 0.0
    return out, score


def find_marker_by_geometry(mask: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Encontra marcador quadrado seguindo o método de teste:
    1) contornos da máscara binária;
    2) filtra áreas [500, 1010];
    3) mantém razão de aspecto em [0.8, 1.2];
    4) escolhe maior ocupação do retângulo envolvente (fill_ratio).

    Args:
        mask: máscara binária numpy (bool ou uint8 0/255) com potencialmente múltiplos contornos.

    Returns:
        (marker_mask, best_contour): máscara do melhor candidato e o contorno OpenCV,
        ou (None, None) se nenhum candidato atender os critérios.
    """
    # Garante formato uint8 para cv2.findContours
    if mask.dtype == np.bool_:
        mask_uint8 = (mask * 255).astype(np.uint8)
    elif mask.dtype == np.uint8:
        mask_uint8 = mask
    else:
        mask_uint8 = (mask * 255).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best_candidate = None
    best_fill = -1.0
    min_area = 300.0
    max_area = 1e10
    min_aspect = 0.8
    max_aspect = 1.5

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Mantém apenas áreas no intervalo especificado.
        if area < min_area or area > max_area:
            continue

        # Retângulo mínimo do contorno
        rect = cv2.minAreaRect(cnt)
        (_, _), (w, h), _ = rect

        if w <= 1 or h <= 1:
            continue

        # Razão de aspecto do retângulo mínimo.
        aspect_ratio = w / (h + 1e-8)
        if aspect_ratio < min_aspect or aspect_ratio > max_aspect:
            continue

        # Quanto o contorno preenche o retângulo
        rect_area = w * h
        fill_ratio = area / (rect_area + 1e-8)

        # Entre candidatos válidos, escolhe o de maior ocupação do retângulo.
        if fill_ratio > best_fill:
            best_fill = fill_ratio
            best_candidate = cnt

    if best_candidate is None:
        return None, None

    # Constrói máscara do melhor candidato
    marker_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(
        marker_mask,
        [best_candidate],
        -1,
        1,
        -1
    )

    return marker_mask, best_candidate


def get_square_component(
    components: List[Component],
) -> Optional[Component]:
    """
    Seleciona o componente de referência (quadrado de marcador) pela sua geometria.

    Usa o método geométrico com filtros explícitos de área/aspecto.
    Não aplica fallback para evitar viés: se não encontrar candidato, retorna None.

    Args:
        components: lista de componentes extraídos da máscara de segmentação.

    Returns:
        Component do marcador, ou None se lista vazia.
    """
    if not components:
        return None

    # Combina máscara de todos os componentes para entrada ao algoritmo
    if len(components) == 1:
        mask_combined = components[0].mask.astype(np.uint8)
    else:
        mask_h, mask_w = components[0].mask.shape
        mask_combined = np.zeros((mask_h, mask_w), dtype=np.uint8)
        for comp in components:
            mask_combined = np.logical_or(mask_combined, comp.mask).astype(np.uint8)

    # Chama algoritmo de detecção de marcador (IC colega)
    marker_mask, best_cnt = find_marker_by_geometry(mask_combined)

    if marker_mask is None or best_cnt is None:
        return None

    # Tenta encontrar qual componente corresponde melhor ao contorno retornado
    marker_mask_bool = marker_mask.astype(bool)
    best_iou = 0.0
    best_comp = None

    for comp in components:
        iou = mask_iou(marker_mask_bool, comp.mask)
        if iou > best_iou:
            best_iou = iou
            best_comp = comp

    # Se encontrou correspondência significativa, retorna
    if best_comp is not None and best_iou > 0.0:
        return best_comp

    return None


def safe_rer(estimated: float, real: float) -> Optional[float]:
    """Calcula erro relativo percentual (RER) com proteção para real <= 0."""
    if real <= 0:
        return 0.0
    if estimated < 0:
        return 0.0
    return abs(estimated - real) / real * 100.0


def csv_number(value: Optional[float]) -> float:
    """Converte valores ausentes para 0.0 no CSV."""
    if value is None:
        return 0.0
    return 0.0 if value < 0 else value


def square_reference_pixels(
    comp: Component,
) -> Tuple[float, float, float]:
    """
    Retorna referências em pixel do quadrado (área, perímetro, lado)
    usando o método fixo:
    - área via contourArea;
    - perímetro via arcLength;
    - lado via perímetro/4.
    """
    comp_u8 = comp.mask.astype(np.uint8)
    contours, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        area_px = comp.area_px
        perim_px = comp.perimeter_px
        side_px = perim_px / 4.0 if perim_px > 0 else 0.0
        return area_px, perim_px, side_px

    cnt = max(contours, key=cv2.contourArea)
    area_px = comp.area_px
    perim_px = comp.perimeter_px
    side_px = perim_px / 4.0 if perim_px > 0 else 0.0
    return area_px, perim_px, side_px


def ensure_dir(path: Path):
    """Cria diretorio (e pais) caso nao exista."""
    path.mkdir(parents=True, exist_ok=True)


def draw_overlay(
    image_path: Optional[Path],
    gt_components: List[Component],
    pred_components: List[Component],
    gt_square: Optional[Component],
    pred_square: Optional[Component],
    matches: List[Tuple[int, int, float, float]],
    overlay_texts: Dict[int, List[str]],
    out_path: Path,
):
    """
    Gera imagem de overlay para depuracao visual:
    - bbox GT (azul),
    - bbox predicao (vermelho/laranja),
    - ligacao entre centroides casados,
    - texto r:/e: dentro da bbox de predicao,
    - IoU abaixo do bloco de texto.
    """
    # Se houver foto original, desenhamos sobre ela; senao, fundo preto.
    if image_path is not None and image_path.exists():
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    else:
        h, w = gt_components[0].mask.shape if gt_components else pred_components[0].mask.shape
        img = np.zeros((h, w, 3), dtype=np.uint8)

    def _draw_text_block_inside_bbox(img, bbox, lines):
        """Desenha bloco de texto dentro da bbox, com auto-ajuste de escala."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        gx, gy, gw, gh = bbox
        if gw <= 8 or gh <= 8 or not lines:
            return 0, 0, 0

        # Comeca grande e reduz automaticamente ate caber na bbox alvo.
        font_scale = 1.30
        min_scale = 0.60
        pad = 5
        line_gap = 4
        text_thickness = 2
        block_w = block_h = line_h = 0
        fits_inside = False
        while True:
            # Mede caixa de texto para decidir se precisa reduzir escala.
            text_sizes = [cv2.getTextSize(line, font, font_scale, text_thickness)[0] for line in lines]
            max_w = max(s[0] for s in text_sizes)
            line_h = max(s[1] for s in text_sizes)
            block_w = max_w + 2 * pad
            block_h = len(lines) * line_h + (len(lines) - 1) * line_gap + 2 * pad
            if (block_w <= gw - 4 and block_h <= gh - 8):
                fits_inside = True
                break
            if font_scale <= min_scale:
                break
            font_scale -= 0.08

        img_h, img_w = img.shape[:2]
        if fits_inside:
            x = gx + 2
            y_top = gy + 2
        else:
            # Se nao couber dentro da bbox, desenha acima (ou abaixo) com escala legivel.
            font_scale = max(min_scale, 0.65)
            text_sizes = [cv2.getTextSize(line, font, font_scale, text_thickness)[0] for line in lines]
            max_w = max(s[0] for s in text_sizes)
            line_h = max(s[1] for s in text_sizes)
            block_w = max_w + 2 * pad
            block_h = len(lines) * line_h + (len(lines) - 1) * line_gap + 2 * pad

            x = max(2, min(gx, img_w - block_w - 2))
            y_top = gy - block_h - 4
            if y_top < 2:
                y_top = gy + gh + 2
            y_top = max(2, min(y_top, img_h - block_h - 2))

        y_bottom = y_top + block_h

        # Caixa de fundo para melhorar contraste do texto sobre qualquer imagem.
        cv2.rectangle(img, (x, y_top), (x + block_w, y_bottom), (35, 35, 35), -1)
        cv2.rectangle(img, (x, y_top), (x + block_w, y_bottom), (170, 170, 170), 1)

        baseline_y = y_top + pad + line_h
        for i, line in enumerate(lines):
            ty = baseline_y + i * (line_h + line_gap)
            cv2.putText(img, line, (x + pad, ty), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, line, (x + pad, ty), font, font_scale, (200, 200, 200), text_thickness, cv2.LINE_AA)

        return block_h, block_w, int(font_scale * 100)

    def _draw_rotated_box(img, comp: Component, color: Tuple[int, int, int], thickness: int = 2):
        """Desenha a menor caixa rotacionada que envolve o componente."""
        comp_u8 = comp.mask.astype(np.uint8)
        contours, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            x, y, w, h = comp.bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
            return

        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) < 3:
            x, y, w, h = comp.bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
            return

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(img, [box], 0, color, thickness)

    match_map = {gi: (pi, iou) for gi, pi, _, iou in matches}

    # BBox GT em azul.
    for gi, g in enumerate(gt_components):
        gx, gy, gw, gh = g.bbox
        cv2.rectangle(img, (gx, gy), (gx + gw, gy + gh), (255, 0, 0), 2)

    # BBox predicao em vermelho; laranja para objetos preditos sem match.
    matched_pred = {pi for _, pi, _, _ in matches}
    for pi, p in enumerate(pred_components):
        px, py, pw, ph = p.bbox
        color = (0, 0, 255) if pi in matched_pred else (0, 120, 220)
        cv2.rectangle(img, (px, py), (px + pw, py + ph), color, 2)

    # Desenha marcador de referencia (quadrado) separadamente para auditoria.
    if gt_square is not None:
        sx, sy, sw, sh = gt_square.bbox
        _draw_rotated_box(img, gt_square, (255, 255, 0), 2)
        cv2.putText(img, "GT marker", (sx, max(14, sy - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 0), 1, cv2.LINE_AA)
    if pred_square is not None:
        sx, sy, sw, sh = pred_square.bbox
        _draw_rotated_box(img, pred_square, (0, 255, 0), 2)
        cv2.putText(img, "Pred marker", (sx, max(28, sy - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 0), 1, cv2.LINE_AA)

    # Linha conectando centroides de pares casados (facilita auditoria visual).
    for gi, pi, _, _ in matches:
        g = gt_components[gi]
        p = pred_components[pi]
        gxy = (int(g.centroid[0]), int(g.centroid[1]))
        pxy = (int(p.centroid[0]), int(p.centroid[1]))
        cv2.line(img, gxy, pxy, (170, 170, 70), 1, cv2.LINE_AA)

    # Texto por ultimo para nao ser coberto por linhas/caixas.
    for gi, g in enumerate(gt_components):
        gx, gy, gw, gh = g.bbox

        lines = overlay_texts.get(gi, [])
        block_h = 0
        block_w = 0
        scale_hint = 62
        if lines:
            # Prioriza desenhar texto dentro da bbox de predicao do objeto casado.
            # Se nao houver match, usa bbox GT.
            if gi in match_map:
                pi, _ = match_map[gi]
                target_bbox = pred_components[pi].bbox
            else:
                target_bbox = g.bbox
            block_h, block_w, scale_hint = _draw_text_block_inside_bbox(img, target_bbox, lines)

        if gi in match_map:
            pi, iou = match_map[gi]
            pbx, pby, pbw, pbh = pred_components[pi].bbox
            # IoU fica abaixo do bloco r:/e:, ainda dentro da bbox vermelha.
            tx = int(pbx + 8)
            ty = int(pby + block_h + 24) if block_h > 0 else int(pby + 24)
            iou_font = max(0.45, min(0.70, scale_hint / 120.0))

            # Clamp para manter o IoU sempre dentro da bbox.
            ty = min(ty, pby + pbh - 6)
            ty = max(ty, pby + 16)

            iou_label = f"IoU={iou:.3f}"
            (iou_w, iou_h), _ = cv2.getTextSize(iou_label, cv2.FONT_HERSHEY_SIMPLEX, iou_font, 2)
            iou_bg_h = iou_h + 8
            iou_bg_w = iou_w + 10
            iou_x = min(tx - 3, pbx + pbw - iou_bg_w - 2)
            iou_x = max(iou_x, pbx + 2)
            iou_y_top = max(min(ty - iou_h - 4, pby + pbh - iou_bg_h - 2), pby + 2)
            iou_y_bottom = iou_y_top + iou_bg_h

            cv2.rectangle(img, (iou_x, iou_y_top), (iou_x + iou_bg_w, iou_y_bottom), (30, 30, 30), -1)
            cv2.rectangle(img, (iou_x, iou_y_top), (iou_x + iou_bg_w, iou_y_bottom), (155, 155, 60), 1)
            cv2.putText(
                img,
                iou_label,
                (iou_x + 5, iou_y_bottom - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                iou_font,
                (210, 210, 120),
                1,
                cv2.LINE_AA,
            )

    cv2.imwrite(str(out_path), img)


def main():
    """
    Fluxo principal do processamento por imagem.

    Para cada arquivo GT:
    1) encontra predicao correspondente;
    2) extrai componentes de folha e square;
    3) resolve fallback com XML quando necessario;
    4) calcula metricas estimadas e RER;
    5) salva CSV e overlays opcionais.
    """
    args = parse_args()
    # Carrega configuracao e resolve caminhos base de entrada/saida.
    config = load_json(Path(args.config))

    image_dir = Path(config.get("image_dir", "")) if config.get("image_dir") else None
    pred_mask_dir = Path(config["pred_mask_dir"])
    gt_mask_dir = Path(config["gt_mask_dir"])
    xml_dir = Path(config["xml_dir"]) if config.get("xml_dir") else None
    results_path = Path(config["results_path"])

    # Paleta usada apenas quando a predição vier pseudo-color.
    class_colors_rgb = config.get("class_colors", [[0, 0, 0], [255, 0, 0], [0, 0, 255]])

    # Mantem a linha do marcador no CSV, como no script YOLO.
    write_square_row = bool(config.get("write_square_row", False))

    # fallback_square_side_cm: usado quando XML nao traz pattern-side.
    fallback_square_side_cm = config.get("fallback_square_side_cm", None)
    fallback_square_side_cm = float(fallback_square_side_cm) if fallback_square_side_cm is not None else None
    xml_coord_mode = str(config.get("xml_coord_mode", "auto")).lower()

    ensure_dir(results_path)
    ensure_dir(results_path / "estimated_metrics")
    if args.save_overlays:
        ensure_dir(results_path / "images" / "matches")

    # Lista de arquivos GT e mapa de predicoes por nome-base.
    gt_mask_dir = Path(config["gt_mask_dir"]) if config.get("gt_mask_dir") else None
    gt_files = sorted([p for p in gt_mask_dir.iterdir() if p.is_file()])
    pred_by_stem = {p.stem: p for p in pred_mask_dir.iterdir() if p.is_file()}

    # Mapa opcional de imagem RGB original para desenhar overlay em cima da foto.
    image_by_stem: Dict[str, Path] = {}
    if image_dir is not None and image_dir.exists():
        for p in image_dir.iterdir():
            if p.is_file():
                image_by_stem[p.stem] = p

    # Arquivo final com metricas por objeto.
    out_csv = results_path / "estimated_metrics" / "estimated_metrics.csv"
    # Escreve CSV do zero a cada execucao.
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Image name", "Leaf index", "image-distance", "pattern-side", "GT object index", "Pred object index", "Class",
            "BBox IoU", "Mask IoU",
            "Real area", "Estimated area (A)", "Estimated area RER (A)", "Estimated area (P)", "Estimated area RER (P)",
            "Real perimeter", "Estimated perimeter (A)", "Estimated perimeter RER (A)", "Estimated perimeter (P)", "Estimated perimeter RER (P)",
            "Real length", "Estimated length (A)", "Estimated length RER (A)", "Estimated length (P)", "Estimated length RER (P)",
            "Real width", "Estimated width (A)", "Estimated width RER (A)", "Estimated width (P)", "Estimated width RER (P)",
        ])

        total_start = time.time()
        processed = 0
        total_images = len(gt_files)

        for gt_path in gt_files:
            img_start = time.time()
            # --- Resolucao de arquivos de entrada por imagem ---
            stem = gt_path.stem
            display_name = gt_path.name
            pred_path = pred_by_stem.get(stem)
            if pred_path is None:
                elapsed = time.time() - img_start
                print(f"[WARN] Missing prediction for {display_name}; skipping. (t={elapsed:.2f}s)")
                continue

            # --- Leitura e normalizacao de tamanho das mascaras ---
            gt_mask = load_mask(gt_path, class_colors_rgb=class_colors_rgb)
            pred_mask = load_mask(pred_path, class_colors_rgb=class_colors_rgb)
            # Garante mesmo tamanho para comparacao pixel a pixel.
            if pred_mask.shape != gt_mask.shape:
                pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

            # --- Extracao de objetos (classe-agnostica) ---
            gt_all = connected_components_all(gt_mask, args.min_object_px, background_id=0)
            pred_all = connected_components_all(pred_mask, args.min_object_px, background_id=0)
            gt_square = get_square_component(gt_all)
            pred_square = get_square_component(pred_all)
            gt_leafs = [c for c in gt_all if gt_square is None or c.comp_id != gt_square.comp_id]
            pred_leafs = [c for c in pred_all if pred_square is None or c.comp_id != pred_square.comp_id]

            # --- Leitura opcional de XML para medidas reais e fallback ---
            xml_leaf_info: List[XmlLeafInfo] = []
            pattern_side_cm: Optional[float] = None
            image_distance_cm: Optional[float] = None
            gt_to_xml: Dict[int, XmlLeafInfo] = {}
            pattern_info: Optional[XmlPatternInfo] = None
            if xml_dir is not None:
                xml_path = xml_dir / f"{stem}.xml"
                image_distance_cm, xml_pattern_side_cm = parse_xml_reference_values(xml_path)
                if xml_coord_mode == "auto":
                    # Testa dois modos de coordenada e escolhe o melhor por score de match.
                    leaves_w, pattern_w = parse_xml_leaf_info(
                        xml_path, gt_mask.shape[1], gt_mask.shape[0], "width")
                    leaves_h, pattern_h = parse_xml_leaf_info(
                        xml_path, gt_mask.shape[1], gt_mask.shape[0], "height")
                    map_w, score_w = map_gt_to_xml(gt_leafs, leaves_w)
                    map_h, score_h = map_gt_to_xml(gt_leafs, leaves_h)

                    if score_h > score_w:
                        xml_leaf_info = leaves_h
                        gt_to_xml = map_h
                        pattern_side_cm = pattern_h
                        pattern_info = parse_xml_pattern_info(
                            xml_path, gt_mask.shape[1], gt_mask.shape[0],
                            "height")
                        ##chosen_mode = "height"
                        #chosen_score = score_h
                    else:
                        xml_leaf_info = leaves_w
                        gt_to_xml = map_w
                        pattern_side_cm = pattern_w
                        pattern_info = parse_xml_pattern_info(
                            xml_path, gt_mask.shape[1], gt_mask.shape[0],
                            "width")
                        #chosen_mode = "width"
                        #chosen_score = score_w
                    #print(
                    #    f"[INFO] {display_name}: xml_coord_mode=auto -> chosen='{chosen_mode}' (match_score={chosen_score:.4f})"
                    #)
                else:
                    if xml_coord_mode not in {"width", "height"}:
                        raise ValueError(
                            "xml_coord_mode must be one of: auto, width, height")
                    xml_leaf_info, pattern_side_cm = parse_xml_leaf_info(
                        xml_path, gt_mask.shape[1], gt_mask.shape[0],
                        xml_coord_mode)
                    gt_to_xml, _ = map_gt_to_xml(gt_leafs, xml_leaf_info)
                    pattern_info = parse_xml_pattern_info(
                        xml_path, gt_mask.shape[1], gt_mask.shape[0],
                        xml_coord_mode)

            # Detection must be independent from XML: do NOT use XML as fallback.
            if gt_square is None or pred_square is None:
                elapsed = time.time() - img_start
                print(f"[WARN] Missing square in GT or pred for {display_name}; skipping image (no XML fallback used). (t={elapsed:.2f}s)")
                continue

            gt_square_area_px, gt_square_perim_px, gt_square_side_px = square_reference_pixels(gt_square)
            pred_square_area_px, pred_square_perim_px, pred_square_side_px = square_reference_pixels(pred_square)

            # pattern_side pode vir do XML ou de fallback no config.
            if pattern_side_cm is None:
                pattern_side_cm = fallback_square_side_cm
            if pattern_side_cm is None:
                print(f"[WARN] No pattern-side available for {display_name}; physical estimates disabled.")

            # Base fisica do marcador: prioriza medidas reais do objeto <pattern>.
            if pattern_info is not None and pattern_info.real_area > 0:
                real_square_area = pattern_info.real_area
                real_square_perimeter = pattern_info.real_perimeter if pattern_info.real_perimeter > 0 else -1.0
                real_square_length = pattern_info.real_length if pattern_info.real_length > 0 else -1.0
                real_square_width = pattern_info.real_width if pattern_info.real_width > 0 else -1.0
            elif pattern_side_cm is not None:
                real_square_area = pattern_side_cm * pattern_side_cm
                real_square_perimeter = pattern_side_cm * 4.0
                real_square_length = pattern_side_cm
                real_square_width = pattern_side_cm
            else:
                real_square_area = -1.0
                real_square_perimeter = -1.0
                real_square_length = -1.0
                real_square_width = -1.0

            if write_square_row:
                square_biou = bbox_iou(gt_square.bbox, pred_square.bbox)
                square_miou = mask_iou(gt_square.mask, pred_square.mask)
                writer.writerow([
                    display_name,
                    0.0,
                    csv_number(image_distance_cm),
                    csv_number(pattern_side_cm),
                    gt_square.comp_id,
                    pred_square.comp_id,
                    "square",
                    square_biou,
                    square_miou,
                    csv_number(real_square_area),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    csv_number(real_square_perimeter),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    csv_number(real_square_length),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    csv_number(real_square_width),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ])

            # Casamento principal GT <-> pred para puxar metrica de cada folha.
            matches = greedy_match(gt_leafs, pred_leafs)
            overlay_texts: Dict[int, List[str]] = {}

            # Keep one row per GT leaf. Unmatched leaves are recovered by bbox/centroid fallback.
            pred_by_gt = match_with_fallback(gt_leafs, pred_leafs)

            for gi, gt_comp in enumerate(gt_leafs):
                # --- Calcula estimativas por objeto GT ---
                pi, biou, iou, match_source = pred_by_gt.get(gi, (-1, 0.0, 0.0, "none"))
                pred_comp = pred_leafs[pi] if pi >= 0 else None

                # Only record CSV rows for GT leaves that actually matched a predicted object.
                if pi < 0 or match_source == "none" or iou <= 0.0:
                    continue

                # Converte de pixel para unidade fisica via regra de tres com quadrado.
                if real_square_area > 0 and gt_square_area_px > 0:
                    est_area_a = (gt_comp.area_px / gt_square_area_px) * real_square_area
                else:
                    est_area_a = -1.0
                if real_square_perimeter > 0 and gt_square_perim_px > 0:
                    est_perim_a = (gt_comp.perimeter_px / gt_square_perim_px) * real_square_perimeter
                else:
                    est_perim_a = -1.0
                if real_square_length > 0 and gt_square_side_px > 0:
                    est_length_a = (gt_comp.length_px / gt_square_side_px) * real_square_length
                else:
                    est_length_a = -1.0
                if real_square_width > 0 and gt_square_side_px > 0:
                    est_width_a = (gt_comp.width_px / gt_square_side_px) * real_square_width
                else:
                    est_width_a = -1.0

                    # (P) usa objeto predito; (A) usa objeto da anotacao (baseline metodo).
                if pred_comp is not None and real_square_area > 0 and pred_square_area_px > 0:
                        est_area_p = (pred_comp.area_px / pred_square_area_px) * real_square_area
                else:
                    est_area_p = -1.0
                if pred_comp is not None and real_square_perimeter > 0 and pred_square_perim_px > 0:
                    est_perim_p = (pred_comp.perimeter_px / pred_square_perim_px) * real_square_perimeter
                else:
                    est_perim_p = -1.0
                if pred_comp is not None and real_square_length > 0 and pred_square_side_px > 0:
                    est_length_p = (pred_comp.length_px / pred_square_side_px) * real_square_length
                else:
                    est_length_p = -1.0
                if pred_comp is not None and real_square_width > 0 and pred_square_side_px > 0:
                    est_width_p = (pred_comp.width_px / pred_square_side_px) * real_square_width
                else:
                    est_width_p = -1.0

                # Dados reais vindos do XML para o objeto casado ao GT.
                xml_match = gt_to_xml.get(gi, None)
                leaf_index = xml_match.leaf_index if xml_match is not None else -1
                real_area = xml_match.real_area if xml_match is not None else -1.0
                real_perimeter = xml_match.real_perimeter if xml_match is not None else -1.0
                real_length = xml_match.real_length if xml_match is not None else -1.0
                real_width = xml_match.real_width if xml_match is not None else -1.0

                # Estrutura final gravada no CSV para auditoria e analise estatistica.
                row = [
                    display_name,
                    leaf_index,
                    csv_number(image_distance_cm),
                    csv_number(pattern_side_cm),
                    gi,
                    pi,
                    "leaf",
                    csv_number(biou),
                    csv_number(iou),
                    csv_number(real_area),
                    csv_number(est_area_a),
                    safe_rer(est_area_a, real_area),
                    csv_number(est_area_p),
                    safe_rer(est_area_p, real_area),
                    csv_number(real_perimeter),
                    csv_number(est_perim_a),
                    safe_rer(est_perim_a, real_perimeter),
                    csv_number(est_perim_p),
                    safe_rer(est_perim_p, real_perimeter),
                    csv_number(real_length),
                    csv_number(est_length_a),
                    safe_rer(est_length_a, real_length),
                    csv_number(est_length_p),
                    safe_rer(est_length_p, real_length),
                    csv_number(real_width),
                    csv_number(est_width_a),
                    safe_rer(est_width_a, real_width),
                    csv_number(est_width_p),
                    safe_rer(est_width_p, real_width),
                ]
                writer.writerow(row)

                # Texto de overlay exibido na imagem para inspecao visual rapida.
                real_area_txt = f"{real_area:.2f}" if real_area > 0 else "NA"
                real_len_txt = f"{real_length:.2f}" if real_length > 0 else "NA"

                # Overlay deve refletir apenas a predicao (P).
                disp_area = est_area_p
                disp_len = est_length_p
                disp_tag = "P"

                est_area_txt = f"{disp_area:.2f}" if disp_area > 0 else "NA"
                est_len_txt = f"{disp_len:.2f}" if disp_len > 0 else "NA"

                overlay_texts[gi] = [
                    f"r: A={real_area_txt} cm2  L={real_len_txt} cm",
                    f"e({disp_tag}/{match_source}): A={est_area_txt} cm2  L={est_len_txt} cm",
                ]

            # Gera imagem de overlay se habilitado por argumento.
            if args.save_overlays:
                overlay_img = image_by_stem.get(stem, None)
                draw_overlay(
                    image_path=overlay_img,
                    gt_components=gt_leafs,
                    pred_components=pred_leafs,
                    gt_square=gt_square,
                    pred_square=pred_square,
                    matches=matches,
                    overlay_texts=overlay_texts,
                    out_path=results_path / "images" / "matches" / display_name,
                )

            # Per-image timing summary
            img_elapsed = time.time() - img_start
            processed += 1
            total_elapsed = time.time() - total_start
            print(f"[INFO] Processed {processed}/{total_images} {display_name} in {img_elapsed:.2f}s (total {total_elapsed/60.0:.2f} min)")

    print(f"Done. Results saved to: {out_csv}")


if __name__ == "__main__":
    main()
