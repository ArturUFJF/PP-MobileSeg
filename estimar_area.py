from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont


# Paleta padrão aplicada em tools/predict.py para modelos com 3 classes.
# Classe 0: fundo (preto)
# Classe 1: folha (vermelho)
# Classe 2: quadrado/pattern (azul)
BACKGROUND_COLOR = (0, 0, 0)
LEAF_COLOR = (255, 0, 0)
SQUARE_COLOR = (0, 0, 255)


@dataclass
class ObjetoXml:
    """Representa um objeto anotado no XML com os dados que interessam ao script."""

    tag: str
    index: int | None
    bbox: tuple[float, float, float, float]
    mask: np.ndarray
    area_real: float | None


@dataclass
class ObjetoPredito:
    """Representa um componente conectado extraído da máscara colorida predita."""

    class_name: str
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    pixel_count: int


def get_pattern_area(xml_path: str | Path) -> float:
    """Lê o lado real do marcador no XML e retorna sua área real."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lado = float(root.find("pattern-side").text)
    return lado * lado


def _collect_coordinate_samples(root: ET.Element) -> tuple[list[float], list[float]]:
    """Coleta amostras de coordenadas cruas do XML para inferir a escala de conversao."""
    x_values: list[float] = []
    y_values: list[float] = []

    for obj in root.findall(".//objects/*"):
        bbox = obj.find("bbox")
        if bbox is not None:
            for tag in ("x", "width"):
                node = bbox.find(tag)
                if node is not None and node.text is not None:
                    x_values.append(float(node.text))
            for tag in ("y", "height"):
                node = bbox.find(tag)
                if node is not None and node.text is not None:
                    y_values.append(float(node.text))

        polygon = obj.find("polygon")
        if polygon is not None:
            i = 1
            while polygon.find(f"x{i}") is not None:
                x_values.append(float(polygon.find(f"x{i}").text))
                y_values.append(float(polygon.find(f"y{i}").text))
                i += 1

    return x_values, y_values


def _get_pattern_bbox_ratio(root: ET.Element, x_scale: float, y_scale: float) -> float | None:
    """Retorna a razao largura/altura do pattern para uma escala candidata."""
    pattern = root.find(".//objects/pattern")
    if pattern is None:
        return None

    bbox = pattern.find("bbox")
    if bbox is not None:
        width = float(bbox.find("width").text) * x_scale
        height = float(bbox.find("height").text) * y_scale
        return (width / height) if height > 0 else None

    polygon = pattern.find("polygon")
    if polygon is None:
        return None

    points: list[tuple[float, float]] = []
    i = 1
    while polygon.find(f"x{i}") is not None:
        points.append(
            (
                float(polygon.find(f"x{i}").text) * x_scale,
                float(polygon.find(f"y{i}").text) * y_scale,
            )
        )
        i += 1

    if not points:
        return None

    _x, _y, width, height = calculate_bounding_box(points)
    return (width / height) if height > 0 else None


def infer_coordinate_scales(root: ET.Element, img_width: int, img_height: int) -> tuple[float, float]:
    """
    Infere fatores de escala para converter coordenadas do XML em pixels.

    Alguns XMLs usam X normalizado por largura, outros por altura.
    A regra abaixo escolhe o fator que minimiza coordenadas fora da imagem.
    """
    x_values, y_values = _collect_coordinate_samples(root)

    y_scale = float(img_height)
    if y_values:
        max_y = max(y_values)
        if max_y > 2.0:
            y_scale = 1.0
        else:
            y_scale = float(img_height)

    x_scale = float(img_width)
    if x_values:
        max_x = max(x_values)
        if max_x > 2.0:
            # Coordenadas ja parecem estar em pixels.
            x_scale = 1.0
        else:
            width_ratio = _get_pattern_bbox_ratio(root, float(img_width), y_scale)
            height_ratio = _get_pattern_bbox_ratio(root, float(img_height), y_scale)

            if width_ratio is not None and height_ratio is not None:
                width_error = abs(width_ratio - 1.0)
                height_error = abs(height_ratio - 1.0)
                x_scale = float(img_height) if height_error < width_error else float(img_width)
            elif max_x > 1.0:
                # Coordenadas X acima de 1 costumam indicar normalizacao por altura.
                x_scale = float(img_height)
            else:
                x_scale = float(img_width)

    return x_scale, y_scale


def calculate_bounding_box(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    """Calcula a bounding box mínima a partir de uma lista de pontos."""
    points_array = np.array(points, dtype=np.float64)
    min_x = float(np.min(points_array[:, 0]))
    min_y = float(np.min(points_array[:, 1]))
    width = float(np.max(points_array[:, 0]) - min_x)
    height = float(np.max(points_array[:, 1]) - min_y)
    return min_x, min_y, width, height


def extract_bounding_box(
    obj: ET.Element,
    x_scale: float,
    y_scale: float,
) -> tuple[float, float, float, float]:
    """
    Extrai a bounding box do XML já em pixels.

    A escala de X e Y e inferida por XML para lidar com anotacoes heterogeneas.
    """
    bbox = obj.find("bbox")
    if bbox is None:
        pontos = extract_points(obj, x_scale, y_scale)
        return calculate_bounding_box(pontos)

    x_min = float(bbox.find("x").text) * x_scale
    y_min = float(bbox.find("y").text) * y_scale
    width = float(bbox.find("width").text) * x_scale
    height = float(bbox.find("height").text) * y_scale
    return x_min, y_min, width, height


def extract_points(
    obj: ET.Element,
    x_scale: float,
    y_scale: float,
) -> list[tuple[float, float]]:
    """Extrai os pontos do polígono do objeto e converte de coordenadas relativas para pixels."""
    points_temp = obj.find("polygon")
    if points_temp is None:
        if obj.find("bbox") is not None:
            raise ValueError("Objeto sem polígono; não é possível gerar máscara real.")
        points_temp = obj

    pontos: list[tuple[float, float]] = []
    i = 1
    while points_temp.find(f"x{i}") is not None:
        x = float(points_temp.find(f"x{i}").text) * x_scale
        y = float(points_temp.find(f"y{i}").text) * y_scale
        pontos.append((x, y))
        i += 1
    return pontos


def get_image_size(xml_path: str | Path) -> tuple[int, int]:
    """Lê largura e altura originais da imagem descritas no XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    width = int(root.find("image-size/width").text)
    height = int(root.find("image-size/height").text)
    return width, height


def draw_corresponding_annotation(obj: ET.Element, img_width: int, img_height: int, x_scale: float, y_scale: float) -> np.ndarray:
    """Gera uma máscara binária da anotação real a partir do polígono do XML."""
    polygon = extract_points(obj, x_scale, y_scale)
    img = Image.new("L", (img_width, img_height), 0)
    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
    return np.array(img, dtype=np.uint8)


def bbox_xywh_to_xyxy(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Converte caixa no formato x, y, largura, altura para x1, y1, x2, y2."""
    x, y, w, h = bbox
    return x, y, x + w, y + h


def compute_bbox_iou(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
) -> float:
    """Calcula IoU entre duas bounding boxes dadas em x, y, largura, altura."""
    ax1, ay1, ax2, ay2 = bbox_xywh_to_xyxy(bbox_a)
    bx1, by1, bx2, by2 = bbox_xywh_to_xyxy(bbox_b)

    x_int_min = max(ax1, bx1)
    y_int_min = max(ay1, by1)
    x_int_max = min(ax2, bx2)
    y_int_max = min(ay2, by2)
    if x_int_max <= x_int_min or y_int_max <= y_int_min:
        return 0.0

    intersection_area = (x_int_max - x_int_min) * (y_int_max - y_int_min)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - intersection_area
    return float(intersection_area / union_area) if union_area > 0 else 0.0


def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Calcula IoU entre duas máscaras binárias de mesmo tamanho."""
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection / union) if union > 0 else 0.0


def load_xml_objects(xml_path: str | Path) -> tuple[list[ObjetoXml], ObjetoXml | None, int, int]:
    """
    Carrega folhas e marcador (`pattern`) a partir do XML.

    O retorno separa folhas e quadrado para facilitar o cálculo da escala real.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width, img_height = get_image_size(xml_path)
    x_scale, y_scale = infer_coordinate_scales(root, img_width, img_height)

    folhas: list[ObjetoXml] = []
    padrao: ObjetoXml | None = None

    for obj in root.findall(".//objects/*"):
        tag = obj.tag.lower()
        bbox = extract_bounding_box(obj, x_scale, y_scale)
        mask = draw_corresponding_annotation(obj, img_width, img_height, x_scale, y_scale)
        index_node = obj.find("index")
        index = int(index_node.text) if index_node is not None else None

        area_real = None
        area_node = obj.find("dimensions/area")
        if area_node is not None:
            area_real = float(area_node.text)

        objeto = ObjetoXml(tag=tag, index=index, bbox=bbox, mask=mask, area_real=area_real)
        if tag == "leaf":
            folhas.append(objeto)
        elif tag == "pattern":
            padrao = objeto

    return folhas, padrao, img_width, img_height


def _extract_connected_components(binary_mask: np.ndarray, class_name: str) -> list[ObjetoPredito]:
    """
    Separa uma máscara binária em componentes conectados.

    Isso é necessário porque no PaddleSeg a cor representa a classe,
    não a instância. Logo, várias folhas aparecem com a mesma cor.
    """
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)

    objetos: list[ObjetoPredito] = []
    for label_idx in range(1, num_labels):
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area <= 0:
            continue

        component_mask = (labels == label_idx).astype(np.uint8)
        objetos.append(
            ObjetoPredito(
                class_name=class_name,
                bbox=(x, y, w, h),
                mask=component_mask,
                pixel_count=area,
            )
        )

    return objetos


def load_predicted_objects(mask_path: str | Path, target_width: int, target_height: int) -> tuple[list[ObjetoPredito], list[ObjetoPredito]]:
    """
    Lê a imagem pseudo-colorida do PaddleSeg e extrai componentes por classe.

    Diferente da versão anterior, aqui não tratamos cada cor como um objeto distinto.
    A cor indica a classe e as instâncias são separadas por conectividade espacial.
    """
    mask_image = Image.open(mask_path).convert("RGB")
    if mask_image.size != (target_width, target_height):
        mask_image = mask_image.resize((target_width, target_height), Image.NEAREST)
    mask_array = np.array(mask_image, dtype=np.uint8)

    leaf_binary = np.all(mask_array == np.array(LEAF_COLOR, dtype=np.uint8), axis=2)
    square_binary = np.all(mask_array == np.array(SQUARE_COLOR, dtype=np.uint8), axis=2)

    folhas_preditas = _extract_connected_components(leaf_binary, "leaf")
    quadrados_preditos = _extract_connected_components(square_binary, "pattern")
    return folhas_preditas, quadrados_preditos


def match_prediction_to_xml(
    objeto_predito: ObjetoPredito,
    objetos_xml: list[ObjetoXml],
) -> tuple[ObjetoXml | None, float, float]:
    """
    Encontra a anotação real correspondente usando IoU de bounding box e máscara.

    Primeiro usa IoU de bbox para filtrar a melhor candidata.
    Em seguida calcula o IoU de máscara da candidata vencedora.
    """
    melhor_objeto: ObjetoXml | None = None
    melhor_bbox_iou = 0.0

    for objeto_xml in objetos_xml:
        bbox_iou = compute_bbox_iou(objeto_predito.bbox, objeto_xml.bbox)
        if bbox_iou > melhor_bbox_iou:
            melhor_bbox_iou = bbox_iou
            melhor_objeto = objeto_xml

    if melhor_objeto is None:
        return None, 0.0, 0.0

    mask_iou = compute_mask_iou(objeto_predito.mask, melhor_objeto.mask)
    return melhor_objeto, melhor_bbox_iou, mask_iou


def ensure_dir(path: str | Path) -> None:
    """Cria diretório se ele ainda não existir."""
    Path(path).mkdir(parents=True, exist_ok=True)


def draw_bbox(draw: ImageDraw.ImageDraw, bbox: tuple[float, float, float, float], color: str) -> None:
    """Desenha bbox no formato x, y, largura, altura."""
    x, y, w, h = bbox
    draw.rectangle([(x, y), (x + w, y + h)], outline=color, width=2)


def process_image(
    image_path: str | Path,
    mask_path: str | Path,
    xml_path: str | Path,
    results_path: str | Path,
    save_masks: bool,
    save_predictions: bool,
) -> None:
    """
    Processa uma imagem:
    - lê a máscara colorida já prevista
    - casa cada cor a uma anotação real do XML
    - calcula área estimada por contagem de pixels
    - gera relatório textual e imagem anotada
    """
    folhas_xml, padrao_xml, img_width, img_height = load_xml_objects(xml_path)
    if padrao_xml is None:
        print(f"Imagem {Path(image_path).name}: XML sem objeto <pattern>, pulando.")
        return

    folhas_preditas, quadrados_preditos = load_predicted_objects(mask_path, img_width, img_height)
    if not folhas_preditas and not quadrados_preditos:
        print(f"Imagem {Path(image_path).name}: nenhuma região prevista nas cores esperadas foi encontrada, pulando.")
        return

    real_square_area = get_pattern_area(xml_path)
    xml_square_pixels = int(np.count_nonzero(padrao_xml.mask))
    square_prediction: ObjetoPredito | None = None
    square_bbox_iou = 0.0
    square_mask_iou = 0.0

    # Tenta descobrir qual componente azul da predição é o marcador real.
    for objeto_predito in quadrados_preditos:
        bbox_iou = compute_bbox_iou(objeto_predito.bbox, padrao_xml.bbox)
        mask_iou = compute_mask_iou(objeto_predito.mask, padrao_xml.mask)
        if mask_iou > square_mask_iou or (mask_iou == square_mask_iou and bbox_iou > square_bbox_iou):
            square_prediction = objeto_predito
            square_bbox_iou = bbox_iou
            square_mask_iou = mask_iou

    if not folhas_preditas:
        print(f"Imagem {Path(image_path).name}: nenhuma folha prevista foi encontrada na máscara, pulando.")
        return

    calibration_source = "xml_pattern"
    calibration_pixels = xml_square_pixels
    if calibration_pixels <= 0:
        if square_prediction is None or square_prediction.pixel_count <= 0:
            print(f"Imagem {Path(image_path).name}: não foi possível obter pixels válidos do quadrado para calibração, pulando.")
            return
        calibration_source = "predicted_pattern"
        calibration_pixels = square_prediction.pixel_count

    area_por_pixel = real_square_area / calibration_pixels
    matched_rows: list[dict[str, float | int | tuple[float, float, float, float] | str]] = []

    # Monta todas as combinacoes folha prevista <-> folha XML para evitar
    # dependencia da ordem dos componentes conectados.
    pair_candidates: list[tuple[float, float, int, int]] = []
    for pred_idx, objeto_predito in enumerate(folhas_preditas):
        for xml_idx, folha_xml in enumerate(folhas_xml):
            bbox_iou = compute_bbox_iou(objeto_predito.bbox, folha_xml.bbox)
            mask_iou = compute_mask_iou(objeto_predito.mask, folha_xml.mask)
            pair_candidates.append((mask_iou, bbox_iou, pred_idx, xml_idx))

    # Prioriza maior IoU de mascara e, em empate, maior IoU de bbox.
    pair_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

    used_pred: set[int] = set()
    used_xml: set[int] = set()

    for mask_iou, bbox_iou, pred_idx, xml_idx in pair_candidates:
        if pred_idx in used_pred or xml_idx in used_xml:
            continue

        # Descarta pares totalmente sem sobreposicao.
        if mask_iou <= 0.0 and bbox_iou <= 0.0:
            continue

        used_pred.add(pred_idx)
        used_xml.add(xml_idx)

        objeto_predito = folhas_preditas[pred_idx]
        melhor_objeto = folhas_xml[xml_idx]
        area_estimada = objeto_predito.pixel_count * area_por_pixel
        area_real = melhor_objeto.area_real if melhor_objeto.area_real is not None else 0.0
        rer = abs(area_estimada - area_real) / area_real * 100 if area_real else 100.0

        matched_rows.append(
            {
                "leaf_index": melhor_objeto.index if melhor_objeto.index is not None else -1,
                "class_name": objeto_predito.class_name,
                "pred_pixels": objeto_predito.pixel_count,
                "bbox_iou": bbox_iou,
                "mask_iou": mask_iou,
                "estimated_area": area_estimada,
                "real_area": area_real,
                "rer": rer,
                "pred_bbox": objeto_predito.bbox,
                "real_bbox": melhor_objeto.bbox,
            }
        )

    ensure_dir(Path(results_path) / "estimated_areas")
    report_path = Path(results_path) / "estimated_areas" / f"{Path(image_path).stem}.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"Image: {Path(image_path).name}\n")
        f.write(f"Prediction mask: {Path(mask_path).name}\n")
        f.write(f"Square real area: {real_square_area:.6f} cm2\n")
        f.write(f"Square XML pixels: {xml_square_pixels}\n")
        f.write(f"Calibration source: {calibration_source}\n")
        if square_prediction is not None:
            f.write(f"Square predicted pixels: {square_prediction.pixel_count}\n")
        else:
            f.write("Square predicted pixels: not found\n")
        f.write(f"Area per pixel: {area_por_pixel:.10f} cm2\n")
        f.write(f"Square bbox IoU: {square_bbox_iou:.4f}\n")
        f.write(f"Square mask IoU: {square_mask_iou:.4f}\n\n")
        f.write("============= Estimated areas =============\n")

        for idx, row in enumerate(matched_rows):
            print(
                f"{Path(image_path).name} | folha XML {row['leaf_index']} | "
                f"bbox IoU {row['bbox_iou']:.2f} | mask IoU {row['mask_iou']:.2f} | "
                f"estimada {row['estimated_area']:.2f} cm2 | real {row['real_area']:.2f} cm2 | "
                f"RER {row['rer']:.2f}%"
            )
            f.write(
                f"Object {idx}: xml leaf index - {row['leaf_index']}, class - {row['class_name']}, "
                f"bbox iou - {row['bbox_iou']:.4f}, segmentation iou - {row['mask_iou']:.4f}, "
                f"estimated area - {row['estimated_area']:.6f} cm2, real area - {row['real_area']:.6f} cm2, "
                f"RER - {row['rer']:.4f}%\n"
            )

    if save_masks:
        ensure_dir(Path(results_path) / "masks")
        Image.open(mask_path).convert("RGB").save(Path(results_path) / "masks" / Path(image_path).name)

    if save_predictions:
        ensure_dir(Path(results_path) / "images")
        base_image = Image.open(image_path).convert("RGB")
        predicted_mask_rgb = Image.open(mask_path).convert("RGB").resize(base_image.size, Image.NEAREST)
        composed = ImageChops.add(base_image, predicted_mask_rgb)
        draw = ImageDraw.Draw(composed)
        fill_color = (0, 0, 0, 255)
        stroke_color = (255, 255, 255, 255)
        font = ImageFont.load_default(size=60)

        for row in matched_rows:
            pred_bbox = row["pred_bbox"]
            real_bbox = row["real_bbox"]
            draw_bbox(draw, pred_bbox, "red")
            draw_bbox(draw, real_bbox, "blue")
            draw.text(
                (pred_bbox[0], pred_bbox[1]),
                (
                    f"leaf {row['leaf_index']}\n"
                    f"e: {row['estimated_area']:.2f} cm2\n"
                    f"r: {row['real_area']:.2f} cm2\n"
                    f"RER: {row['rer']:.2f}%"
                ),
                fill=fill_color,
                stroke_fill=stroke_color,
                stroke_width=2,
                font=font,
            )

        if square_prediction is not None:
            draw_bbox(draw, square_prediction.bbox, "red")
        draw_bbox(draw, padrao_xml.bbox, "blue")
        draw.text(
            (padrao_xml.bbox[0], padrao_xml.bbox[1]),
            f"quadrado\nr: {real_square_area:.2f} cm2\ncal: {calibration_source}",
            fill=fill_color,
            stroke_fill=stroke_color,
            stroke_width=2,
            font=font,
        )
        composed.save(Path(results_path) / "images" / Path(image_path).name)


def load_config(config_path: str | Path) -> dict[str, str]:
    """Carrega o arquivo de configuração JSON com os diretórios de entrada e saída."""
    with Path(config_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def find_prediction_mask(image_name: str, prediction_dir: str | Path) -> Path | None:
    """
    Procura a máscara predita correspondente à imagem.

    Aceita extensões comuns e mantém o mesmo nome base da imagem original.
    """
    base_name = Path(image_name).stem
    for extension in (".png", ".jpg", ".jpeg", ".bmp"):
        candidate = Path(prediction_dir) / f"{base_name}{extension}"
        if candidate.exists():
            return candidate
    return None


def build_prediction_index(prediction_dir: str | Path) -> dict[str, Path]:
    """
    Indexa mascaras previstas por stem (case-insensitive), buscando recursivamente.

    Se houver duplicatas de stem, fica com o primeiro caminho encontrado.
    """
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    index: dict[str, Path] = {}
    for path in Path(prediction_dir).rglob("*"):
        if not path.is_file() or path.suffix.lower() not in supported_extensions:
            continue
        key = path.stem.lower()
        if key not in index:
            index[key] = path
    return index


def main() -> int:
    """
    Executa o processamento em lote usando máscaras coloridas já geradas por um modelo.

    Espera encontrar no config.json:
    - image_dir: pasta com imagens originais
    - xml_dir: pasta com XMLs
    - prediction_dir: pasta com máscaras coloridas previstas
    - results_path: pasta de saída
    """
    parser = argparse.ArgumentParser(description="Estimativa de área a partir de máscaras coloridas já previstas.")
    parser.add_argument("-c", "--config", default="./config.json", help="Caminho do arquivo config.json")
    parser.add_argument("-m", "--save-masks", action="store_true", help="Salvar as máscaras coloridas copiadas para a pasta de resultados")
    parser.add_argument("-p", "--save-predictions", action="store_true", help="Salvar as imagens finais com caixas e áreas anotadas")
    args = parser.parse_args()

    config = load_config(args.config)
    image_dir = Path(config["image_dir"])
    xml_dir = Path(config["xml_dir"])
    prediction_dir = Path(config["prediction_dir"])
    results_path = Path(config["results_path"])

    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_index = {
        p.stem.lower(): p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in supported_extensions
    }
    prediction_index = build_prediction_index(prediction_dir)

    if not prediction_index:
        print("Nenhuma máscara prevista encontrada no diretório de predição.")
        return 0

    processed = 0
    missing_image = 0
    missing_xml = 0

    for stem_key, mask_path in sorted(prediction_index.items(), key=lambda item: item[0]):
        image_path = image_index.get(stem_key)
        if image_path is None:
            missing_image += 1
            print(f"Imagem original não encontrada para máscara {mask_path.name}, pulando.")
            continue

        xml_path = xml_dir / f"{image_path.stem}.xml"
        if not xml_path.exists():
            missing_xml += 1
            print(f"XML não encontrado para {image_path.name}, pulando.")
            continue

        process_image(
            image_path=image_path,
            mask_path=mask_path,
            xml_path=xml_path,
            results_path=results_path,
            save_masks=args.save_masks,
            save_predictions=args.save_predictions,
        )
        processed += 1

    print(
        f"Resumo: processadas {processed} imagens a partir de {len(prediction_index)} máscaras. "
        f"Sem imagem correspondente: {missing_image}. Sem XML correspondente: {missing_xml}."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())