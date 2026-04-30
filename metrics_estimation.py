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
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]


@dataclass
class XmlLeafInfo:
    leaf_index: int
    mask: np.ndarray
    real_area: float
    real_perimeter: float
    real_length: float


@dataclass
class XmlPatternInfo:
    mask: np.ndarray
    real_area: float
    real_perimeter: float
    real_length: float
    pattern_side_cm: Optional[float]


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
        area = float(stats[comp_id, cv2.CC_STAT_AREA])
        if area < min_px:
            continue

        x = int(stats[comp_id, cv2.CC_STAT_LEFT])
        y = int(stats[comp_id, cv2.CC_STAT_TOP])
        w = int(stats[comp_id, cv2.CC_STAT_WIDTH])
        h = int(stats[comp_id, cv2.CC_STAT_HEIGHT])

        comp_mask = labels == comp_id
        comp_u8 = comp_mask.astype(np.uint8)
        # Extrai contornos para calcular perimetro e comprimento.
        contours, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        perimeter = 0.0
        length = 0.0
        for cnt in contours:
            perimeter += float(cv2.arcLength(cnt, True))
            # Se houver pontos suficientes, usamos PCA para estimar eixo principal.
            if len(cnt) >= 5:
                pts = cnt.reshape(-1, 2).astype(np.float32)
                mean, eigenvectors = cv2.PCACompute(pts, mean=None)
                proj = cv2.PCAProject(pts, mean, eigenvectors)
                min_proj = np.min(proj, axis=0)
                max_proj = np.max(proj, axis=0)
                cand = float(max(max_proj[0] - min_proj[0], max_proj[1] - min_proj[1]))
                length = max(length, cand)
            else:
                # Fallback geometrico para contornos muito pequenos.
                (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
                length = max(length, float(max(rw, rh)))

        cx, cy = centroids[comp_id]
        comps.append(
            Component(
                comp_id=comp_id,
                mask=comp_mask,
                area_px=area,
                perimeter_px=perimeter,
                length_px=length,
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
    area = float(comp_u8.sum())
    if area <= 0:
        return None

    # Bounding box por extremos dos pixels ativos.
    ys, xs = np.where(comp_u8 > 0)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    bbox = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
    centroid = (float(xs.mean()), float(ys.mean()))

    # Extrai forma para perimetro/comprimento, igual ao fluxo normal.
    contours, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    perimeter = 0.0
    length = 0.0
    for cnt in contours:
        perimeter += float(cv2.arcLength(cnt, True))
        if len(cnt) >= 5:
            pts = cnt.reshape(-1, 2).astype(np.float32)
            mean, eigenvectors = cv2.PCACompute(pts, mean=None)
            proj = cv2.PCAProject(pts, mean, eigenvectors)
            min_proj = np.min(proj, axis=0)
            max_proj = np.max(proj, axis=0)
            cand = float(max(max_proj[0] - min_proj[0], max_proj[1] - min_proj[1]))
            length = max(length, cand)
        else:
            (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
            length = max(length, float(max(rw, rh)))

    return Component(
        comp_id=1,
        mask=mask_bool,
        area_px=area,
        perimeter_px=perimeter,
        length_px=length,
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


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Calcula IoU entre duas mascaras binarias."""
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
            iou = mask_iou(g.mask, p.mask)
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
                best_bbox_mask_iou = mask_iou(gt_comp.mask, pred_comp.mask)
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
            mask_iou(gt_comp.mask, pred_comp.mask),
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

    pattern_side = root.find("pattern-side")
    pattern_side_cm = float(pattern_side.text) if pattern_side is not None else None

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

        area = float(area_txt) if area_txt != "NAva" else -1.0
        perimeter = float(per_txt) if per_txt != "NAva" else -1.0
        length = float(len_txt) if len_txt != "NAva" else -1.0

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
                real_area=area,
                real_perimeter=perimeter,
                real_length=length,
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

    pattern_side_node = root.find("pattern-side")
    pattern_side_cm = None
    if pattern_side_node is not None and pattern_side_node.text is not None:
        try:
            pattern_side_cm = float(pattern_side_node.text)
        except ValueError:
            pattern_side_cm = None

    points = get_xml_points(pattern_node, width, height, coord_mode)
    if not points:
        return None

    real_area = -1.0
    real_perimeter = -1.0
    real_length = -1.0
    dims = pattern_node.find("dimensions")
    if dims is not None:
        area_txt = dims.find("area").text if dims.find("area") is not None else "-1"
        per_txt = dims.find("perimeter").text if dims.find("perimeter") is not None else "-1"
        len_txt = dims.find("length").text if dims.find("length") is not None else "-1"
        real_area = float(area_txt) if area_txt != "NAva" else -1.0
        real_perimeter = float(per_txt) if per_txt != "NAva" else -1.0
        real_length = float(len_txt) if len_txt != "NAva" else -1.0
    elif pattern_side_cm is not None:
        real_area = pattern_side_cm * pattern_side_cm
        real_perimeter = pattern_side_cm * 4.0
        real_length = pattern_side_cm

    return XmlPatternInfo(
        mask=polygon_to_mask(points, width, height),
        real_area=real_area,
        real_perimeter=real_perimeter,
        real_length=real_length,
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
            iou = mask_iou(g.mask, xobj.mask)
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


def is_square_by_geometry(comp: Component, aspect_threshold: float = 0.70,
                          perim_tolerance: float = 0.30) -> float:
    """
    Calcula um score de "quadrado-ness" para um componente.
    
    Heurísticas:
    - aspecto_ratio: min(w,h) / max(w,h) proximo de 1 => mais quadrado
    - perimetro esperado para quadrado perfeito: 4*sqrt(area)
    - se real proximo de esperado => mais quadrado
    
    Retorna score entre 0 e 1: 1 = quadrado perfeito, 0 = nao e quadrado.
    """
    x, y, w, h = comp.bbox
    if w <= 0 or h <= 0:
        return 0.0

    # Area relativa no frame completo: marcador tende a ser pequeno.
    img_area = float(comp.mask.size) if comp.mask is not None else 0.0
    area_ratio = (comp.area_px / img_area) if img_area > 0 else 1.0

    # Aspecto ratio: quanto mais proximo de 1, melhor.
    aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0.0
    if aspect < aspect_threshold:
        return 0.0
    aspect_score = aspect

    # Bbox fill: quadrado tende a preencher bem a bbox; folhas tem recortes/concavidades.
    bbox_area = float(w * h)
    fill_ratio = (comp.area_px / bbox_area) if bbox_area > 0 else 0.0
    fill_score = max(0.0, min(fill_ratio, 1.0))

    # Verifica numero de vertices aproximados no contorno principal.
    comp_u8 = comp.mask.astype(np.uint8)
    contours, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True) if peri > 0 else cnt
        vertex_count = len(approx)
    else:
        vertex_count = 0
    if vertex_count == 4:
        vertex_score = 1.0
    elif vertex_count in (5, 6):
        vertex_score = 0.65
    elif vertex_count in (3, 7):
        vertex_score = 0.35
    else:
        vertex_score = 0.10

    # Perímetro esperado para um quadrado de area A: 4*sqrt(A)
    expected_perim = 4.0 * math.sqrt(comp.area_px)
    if expected_perim <= 0:
        return 0.0

    actual_perim = comp.perimeter_px
    perim_ratio = actual_perim / expected_perim

    # Score diminui conforme desvia de 1.0, com tolerancia.
    perim_score = 1.0 - min(abs(perim_ratio - 1.0), 1.0)
    if perim_ratio > 1.0 + perim_tolerance or perim_ratio < 1.0 - perim_tolerance:
        perim_score *= 0.5

    # Prior de tamanho: marcador costuma ocupar pequena fracao da imagem.
    if area_ratio <= 0.015:
        size_score = 1.0
    elif area_ratio >= 0.10:
        size_score = 0.0
    else:
        size_score = max(0.0, 1.0 - (area_ratio - 0.015) / (0.10 - 0.015))

    # Score final: combina forma + contorno + tamanho esperado de marcador.
    score = (
        0.30 * aspect_score +
        0.20 * fill_score +
        0.20 * vertex_score +
        0.15 * perim_score +
        0.15 * size_score
    )
    return score


def get_square_component(
    components: List[Component],
    min_score: float = 0.5,
    fallback_policy: str = "largest",
) -> Optional[Component]:
    """
    Seleciona o componente de referencia para o quadrado pela sua geometria.
    
    Prioriza componentes que:
    1. Tem aspecto ratio proximo de 1 (width ≈ height)
    2. Tem perimetro coerente com uma forma quadrada
    
    Se nenhum candidato "quadrado" for encontrado, faz fallback para o maior.
    """
    if not components:
        return None

    # Tenta encontrar componentes com geometria de quadrado.
    scored = [(is_square_by_geometry(c), i, c) for i, c in enumerate(components)]
    # Em empate de score, prioriza o menor componente (marcador tende a ser menor que folhas).
    scored.sort(key=lambda x: (x[0], -x[2].area_px), reverse=True)

    # Se o melhor candidato tem score razoavel, retorna ele.
    if scored[0][0] >= min_score:
        return scored[0][2]

    # Fallback configuravel: em classe unica geralmente faz sentido priorizar menor area.
    if fallback_policy == "smallest":
        return min(components, key=lambda c: c.area_px)
    return max(components, key=lambda c: c.area_px)


def select_single_class_marker(
    components: List[Component],
    reference_mask: Optional[np.ndarray],
    min_square_score: float,
    fallback_policy: str,
) -> Optional[Component]:
    """
    Seleciona marcador no cenário de classe única (folha e quadrado compartilham id).

    Estratégia alinhada ao fluxo YOLO funcional:
    1) quando existir máscara de referência no XML (<pattern>), ancora a seleção nela;
    2) sem referência, usa score geométrico de quadrado;
    3) se o score for fraco, aplica fallback configurável (smallest/largest).
    """
    if not components:
        return None

    if reference_mask is not None:
        by_ref = get_component_by_reference_mask(components, reference_mask)
        if by_ref is not None:
            return by_ref

    return get_square_component(
        components,
        min_score=min_square_score,
        fallback_policy=fallback_policy,
    )


def get_component_by_reference_mask(components: List[Component],
                                    ref_mask: np.ndarray) -> Optional[Component]:
    """
    Seleciona o componente mais compatível com uma máscara de referência.

    Estratégia:
    1) maximiza IoU com a máscara de referência;
    2) se IoU for zero para todos, usa menor distância entre centroides.
    """
    if not components:
        return None

    best_by_iou = None
    best_iou = -1.0
    for comp in components:
        iou = mask_iou(comp.mask, ref_mask)
        if iou > best_iou:
            best_iou = iou
            best_by_iou = comp

    if best_by_iou is not None and best_iou > 0.0:
        return best_by_iou

    ref_comp = component_from_binary_mask(ref_mask)
    if ref_comp is None:
        return best_by_iou

    rx, ry = ref_comp.centroid
    return min(
        components,
        key=lambda c: (c.centroid[0] - rx) ** 2 + (c.centroid[1] - ry) ** 2,
    )


def safe_rer(estimated: float, real: float) -> float:
    """Calcula erro relativo percentual (RER) com protecao para real <= 0."""
    if real <= 0:
        return -1.0
    return abs(estimated - real) / real * 100.0


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

    # IDs de classe esperados pelo dataset de segmentacao.
    class_ids = config.get("class_ids", {})
    leaf_id = int(class_ids.get("leaf", 1))
    square_id = int(class_ids.get("square", 2))
    class_colors_rgb = config.get("class_colors", [[0, 0, 0], [255, 0, 0], [0, 0, 255]])

    mode = str(config.get("mode", "auto")).lower()
    if mode not in {"auto", "single_class", "multi_class"}:
        raise ValueError("config.mode must be one of: auto, single_class, multi_class")
    if mode == "single_class":
        use_single_class = True
    elif mode == "multi_class":
        use_single_class = False
    else:
        use_single_class = (leaf_id == square_id)

    # Parametros do cenário single-class (alinhado ao fluxo YOLO funcional).
    single_class_use_xml_pattern = bool(config.get("single_class_use_xml_pattern", True))
    single_class_marker_min_score = float(config.get("single_class_marker_min_score", 0.5))
    single_class_marker_fallback = str(
        config.get("single_class_marker_fallback", "smallest")
    ).lower()
    if single_class_marker_fallback not in {"smallest", "largest"}:
        raise ValueError("single_class_marker_fallback must be 'smallest' or 'largest'")

    # Mantem a linha do marcador no CSV, como no script YOLO.
    write_square_row = bool(config.get("write_square_row", True))

    # fallback_square_side_cm: usado quando XML nao traz pattern-side.
    fallback_square_side_cm = config.get("fallback_square_side_cm", None)
    fallback_square_side_cm = float(fallback_square_side_cm) if fallback_square_side_cm is not None else None
    xml_coord_mode = str(config.get("xml_coord_mode", "auto")).lower()

    ensure_dir(results_path)
    ensure_dir(results_path / "estimated_metrics")
    if args.save_overlays:
        ensure_dir(results_path / "images" / "matches")

    # Lista de arquivos GT e mapa de predicoes por nome-base.
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
            "Image name", "Leaf index", "GT object index", "Pred object index", "Class",
            "BBox IoU", "Mask IoU",
            "Real area", "Estimated area (A)", "Estimated area RER (A)", "Estimated area (P)", "Estimated area RER (P)",
            "Real perimeter", "Estimated perimeter (A)", "Estimated perimeter RER (A)", "Estimated perimeter (P)", "Estimated perimeter RER (P)",
            "Real length", "Estimated length (A)", "Estimated length RER (A)", "Estimated length (P)", "Estimated length RER (P)",
        ])

        for gt_path in gt_files:
            # --- Resolucao de arquivos de entrada por imagem ---
            stem = gt_path.stem
            pred_path = pred_by_stem.get(stem)
            if pred_path is None:
                print(f"[WARN] Missing prediction for {gt_path.name}; skipping.")
                continue

            # --- Leitura e normalizacao de tamanho das mascaras ---
            gt_mask = load_mask(gt_path, class_colors_rgb=class_colors_rgb)
            pred_mask = load_mask(pred_path, class_colors_rgb=class_colors_rgb)
            # Garante mesmo tamanho para comparacao pixel a pixel.
            if pred_mask.shape != gt_mask.shape:
                pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

            # --- Extracao de objetos (folha e quadrado) ---
            gt_all: List[Component] = []
            pred_all: List[Component] = []
            # Classe única: folhas e marcador compartilham a mesma classe.
            if use_single_class:
                gt_all = connected_components(gt_mask, leaf_id, args.min_object_px)
                pred_all = connected_components(pred_mask, leaf_id, args.min_object_px)
                gt_square = None
                pred_square = None
                gt_leafs = list(gt_all)
                pred_leafs = list(pred_all)
            else:
                gt_leafs = connected_components(gt_mask, leaf_id, args.min_object_px)
                pred_leafs = connected_components(pred_mask, leaf_id, args.min_object_px)
                gt_squares = connected_components(gt_mask, square_id, args.min_object_px)
                pred_squares = connected_components(pred_mask, square_id, args.min_object_px)

                gt_square = get_square_component(gt_squares)
                pred_square = get_square_component(pred_squares)

            # --- Leitura opcional de XML para medidas reais e fallback ---
            xml_leaf_info: List[XmlLeafInfo] = []
            pattern_side_cm: Optional[float] = None
            gt_to_xml: Dict[int, XmlLeafInfo] = {}
            pattern_info: Optional[XmlPatternInfo] = None
            if xml_dir is not None:
                xml_path = xml_dir / f"{stem}.xml"
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
                        chosen_mode = "height"
                        chosen_score = score_h
                    else:
                        xml_leaf_info = leaves_w
                        gt_to_xml = map_w
                        pattern_side_cm = pattern_w
                        pattern_info = parse_xml_pattern_info(
                            xml_path, gt_mask.shape[1], gt_mask.shape[0],
                            "width")
                        chosen_mode = "width"
                        chosen_score = score_w
                    print(
                        f"[INFO] {gt_path.name}: xml_coord_mode=auto -> chosen='{chosen_mode}' (match_score={chosen_score:.4f})"
                    )
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

            # --- Fallback do quadrado de referencia usando <pattern> do XML ---
            ref_square = None
            if pattern_info is not None:
                ref_square = component_from_binary_mask(pattern_info.mask)
                if pattern_side_cm is None and pattern_info.pattern_side_cm is not None:
                    pattern_side_cm = pattern_info.pattern_side_cm

            # Em classe única, selecionamos marcador com prioridade para ancoragem no pattern.
            if use_single_class:
                reference_mask = ref_square.mask if (single_class_use_xml_pattern and ref_square is not None) else None
                gt_square = select_single_class_marker(
                    gt_all,
                    reference_mask=reference_mask,
                    min_square_score=single_class_marker_min_score,
                    fallback_policy=single_class_marker_fallback,
                )
                pred_square = select_single_class_marker(
                    pred_all,
                    reference_mask=reference_mask,
                    min_square_score=single_class_marker_min_score,
                    fallback_policy=single_class_marker_fallback,
                )

                # Reconstroi listas de folhas removendo o marcador final escolhido.
                gt_leafs = [
                    c for c in gt_all
                    if gt_square is None or c.comp_id != gt_square.comp_id
                ]
                pred_leafs = [
                    c for c in pred_all
                    if pred_square is None or c.comp_id != pred_square.comp_id
                ]
                # Atualiza mapeamento GT->XML após o refinamento da separação folha/marcador.
                if xml_leaf_info:
                    gt_to_xml, _ = map_gt_to_xml(gt_leafs, xml_leaf_info)

            # Em multi-classe, se faltar quadrado nas mascaras, usa pattern do XML como referencia.
            if not use_single_class and gt_square is None and ref_square is not None:
                gt_square = ref_square
                print(f"[INFO] {gt_path.name}: using XML pattern as GT square reference.")
            if not use_single_class and pred_square is None and ref_square is not None:
                pred_square = ref_square
                print(f"[INFO] {gt_path.name}: using XML pattern as prediction square reference.")

            if gt_square is None or pred_square is None:
                print(
                    f"[WARN] Missing square in GT/pred and XML pattern fallback unavailable for {gt_path.name}; skipping image."
                )
                continue

            # pattern_side pode vir do XML ou de fallback no config.
            if pattern_side_cm is None:
                pattern_side_cm = fallback_square_side_cm
            if pattern_side_cm is None:
                print(f"[WARN] No pattern-side available for {gt_path.name}; physical estimates disabled.")

            # Base fisica do marcador: prioriza medidas reais do objeto <pattern>.
            if pattern_info is not None and pattern_info.real_area > 0:
                real_square_area = pattern_info.real_area
                real_square_perimeter = pattern_info.real_perimeter if pattern_info.real_perimeter > 0 else -1.0
                real_square_length = pattern_info.real_length if pattern_info.real_length > 0 else -1.0
            elif pattern_side_cm is not None:
                real_square_area = pattern_side_cm * pattern_side_cm
                real_square_perimeter = pattern_side_cm * 4.0
                real_square_length = pattern_side_cm
            else:
                real_square_area = -1.0
                real_square_perimeter = -1.0
                real_square_length = -1.0

            if write_square_row:
                square_biou = bbox_iou(gt_square.bbox, pred_square.bbox)
                square_miou = mask_iou(gt_square.mask, pred_square.mask)
                writer.writerow([
                    gt_path.name,
                    -1,
                    gt_square.comp_id,
                    pred_square.comp_id,
                    "square",
                    square_biou,
                    square_miou,
                    real_square_area,
                    -1.0,
                    0.0,
                    -1.0,
                    0.0,
                    real_square_perimeter,
                    -1.0,
                    0.0,
                    -1.0,
                    0.0,
                    real_square_length,
                    -1.0,
                    0.0,
                    -1.0,
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

                # Converte de pixel para unidade fisica via regra de tres com quadrado.
                if real_square_area > 0 and gt_square.area_px > 0:
                    est_area_a = (gt_comp.area_px / gt_square.area_px) * real_square_area
                else:
                    est_area_a = -1.0
                if real_square_perimeter > 0 and gt_square.perimeter_px > 0:
                    est_perim_a = (gt_comp.perimeter_px / gt_square.perimeter_px) * real_square_perimeter
                else:
                    est_perim_a = -1.0
                if real_square_length > 0 and gt_square.length_px > 0:
                    est_length_a = (gt_comp.length_px / gt_square.length_px) * real_square_length
                else:
                    est_length_a = -1.0

                    # (P) usa objeto predito; (A) usa objeto da anotacao (baseline metodo).
                if pred_comp is not None and real_square_area > 0 and pred_square.area_px > 0:
                        est_area_p = (pred_comp.area_px / pred_square.area_px) * real_square_area
                else:
                    est_area_p = -1.0
                if pred_comp is not None and real_square_perimeter > 0 and pred_square.perimeter_px > 0:
                    est_perim_p = (pred_comp.perimeter_px / pred_square.perimeter_px) * real_square_perimeter
                else:
                    est_perim_p = -1.0
                if pred_comp is not None and real_square_length > 0 and pred_square.length_px > 0:
                    est_length_p = (pred_comp.length_px / pred_square.length_px) * real_square_length
                else:
                    est_length_p = -1.0

                # Dados reais vindos do XML para o objeto casado ao GT.
                xml_match = gt_to_xml.get(gi, None)
                leaf_index = xml_match.leaf_index if xml_match is not None else -1
                real_area = xml_match.real_area if xml_match is not None else -1.0
                real_perimeter = xml_match.real_perimeter if xml_match is not None else -1.0
                real_length = xml_match.real_length if xml_match is not None else -1.0

                # Estrutura final gravada no CSV para auditoria e analise estatistica.
                row = [
                    gt_path.name,
                    leaf_index,
                    gi,
                    pi,
                    "leaf",
                    biou,
                    iou,
                    real_area,
                    est_area_a,
                    safe_rer(est_area_a, real_area),
                    est_area_p,
                    safe_rer(est_area_p, real_area),
                    real_perimeter,
                    est_perim_a,
                    safe_rer(est_perim_a, real_perimeter),
                    est_perim_p,
                    safe_rer(est_perim_p, real_perimeter),
                    real_length,
                    est_length_a,
                    safe_rer(est_length_a, real_length),
                    est_length_p,
                    safe_rer(est_length_p, real_length),
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
                    out_path=results_path / "images" / "matches" / gt_path.name,
                )

    print(f"Done. Results saved to: {out_csv}")


if __name__ == "__main__":
    main()
