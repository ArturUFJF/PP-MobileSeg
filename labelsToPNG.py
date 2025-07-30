import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import re

# Caminhos
xml_dir = "data/dimensionamento_foliar/masks"  # onde estão os arquivos .xml
image_dir = "data/dimensionamento_foliar/images"  # onde estão as imagens .jpg ou .png
output_mask_dir = "data/dimensionamento_foliar/labels"  # onde as máscaras serão salvas

os.makedirs(output_mask_dir, exist_ok=True)

def normalize_filename(filename):
    """Normaliza nome do arquivo: minusculas, substitui espaços por hifen, remove caracteres especiais"""
    # Remove extensão
    name, ext = os.path.splitext(filename)
    
    # Converte para minúsculas
    name = name.lower()
    
    # Substitui espaços, underscores e outros separadores por hífen
    name = re.sub(r'[\s_]+', '-', name)
    
    # Remove caracteres especiais, mantendo apenas letras, números e hífens
    name = re.sub(r'[^a-z0-9\-]', '', name)
    
    # Remove hífens duplos ou múltiplos
    name = re.sub(r'-+', '-', name)
    
    # Remove hífens no início e fim
    name = name.strip('-')
    
    return name + ext

def rename_files_in_directory(directory):
    """Renomeia todos os arquivos em um diretório para seguir o padrão normalizado"""
    if not os.path.exists(directory):
        print(f"Diretorio nao existe: {directory}")
        return
    
    renamed_count = 0
    for filename in os.listdir(directory):
        old_path = os.path.join(directory, filename)
        if os.path.isfile(old_path):
            new_filename = normalize_filename(filename)
            new_path = os.path.join(directory, new_filename)
            
            if filename != new_filename:
                # Verifica se o novo nome já existe
                if os.path.exists(new_path):
                    print(f"AVISO: {new_filename} já existe. Pulando {filename}")
                    continue
                
                os.rename(old_path, new_path)
                print(f"Renomeado: {filename} -> {new_filename}")
                renamed_count += 1
    
    print(f"Total de arquivos renomeados em {directory}: {renamed_count}")

# Primeiro, normaliza nomes dos arquivos em todos os diretórios
print("=== NORMALIZANDO NOMES DOS ARQUIVOS ===")
rename_files_in_directory(xml_dir)
rename_files_in_directory(image_dir)

print("\n=== INICIANDO CONVERSAO XML PARA PNG ===")

os.makedirs(output_mask_dir, exist_ok=True)

print(f"Processando XMLs do diretorio: {xml_dir}")
print(f"Buscando imagens no diretorio: {image_dir}")
print(f"Salvando mascaras em: {output_mask_dir}")

xml_files = [f for f in os.listdir(xml_dir) if f.endswith(".xml")]
print(f"Encontrados {len(xml_files)} arquivos XML")

def extract_polygon_coords(obj, width, height):
    """Extrai e ajusta coordenadas de polígono do XML para o tamanho da imagem"""
    coords = []
    max_x = 0
    max_y = 0

    # Extrai todas as coordenadas
    for i in range(1, 1000):
        x_tag = obj.find(f'x{i}')
        y_tag = obj.find(f'y{i}')
        if x_tag is None or y_tag is None:
            break

        try:
            x_val = int(x_tag.text)
            y_val = int(y_tag.text)
            coords.append((x_val, y_val))
            max_x = max(max_x, x_val)
            max_y = max(max_y, y_val)
        except (ValueError, TypeError):
            continue

    if not coords:
        return np.array([], dtype=np.int32)

    coords_np = np.array([coords], dtype=np.int32)

    # Verifica se precisa escalar
    if max_x > width or max_y > height:
        scale_x = width / (max_x + 1)
        scale_y = height / (max_y + 1)
        scale = min(scale_x, scale_y)  # mesma escala para manter proporção
        coords_np = (coords_np * scale).astype(np.int32)

    return coords_np


def validate_polygon(coords, width, height):
    """Verifica se todas as coordenadas do poligono estao dentro da imagem"""
    if coords.size == 0:
        return False
    for x, y in coords[0]:
        if not (0 <= x < width and 0 <= y < height):
            return False
    return True

def process_xml_to_mask(xml_path, image_dir, output_mask_dir):
    """Processa um arquivo XML e gera a mascara PNG correspondente"""
    filename = os.path.basename(xml_path)

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Erro ao fazer parse do XML {filename}: {e}")
        return False
    except Exception as e:
        print(f"Erro inesperado ao processar {filename}: {e}")
        return False

    # Nome da imagem associado
    filename_element = root.find("filename")
    if filename_element is None or filename_element.text is None:
        print(f"Elemento 'filename' nao encontrado em {filename}")
        return False

    original_image_name = filename_element.text.strip()
    base_name = normalize_filename(original_image_name)
    base_name = re.sub(r'(_mask|-mask)?\.(jpg|jpeg|png)$', '', base_name)

    # Tenta encontrar a imagem correspondente
    image_path = None
    for ext in [".jpg", ".jpeg", ".png"]:
        candidate = os.path.join(image_dir, base_name + ext)
        if os.path.exists(candidate):
            image_path = candidate
            break

    if image_path is None:
        print(f"Imagem nao encontrada para {filename}. Tentado base: {base_name}")
        return False

    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro ao abrir: {image_path}")
        return False

    height, width = img.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    # Processa objetos (folhas e quadrado)
    object_element = root.find("object")
    if object_element is None:
        print(f"Elemento <object> nao encontrado em {filename}")
        return False

    objects_processed = 0

    # Folhas (classe 1)
    leaf_elements = [child for child in object_element if child.tag.startswith("leaf")]
    for leaf in leaf_elements:
        try:
            coords = extract_polygon_coords(leaf, width, height)
            if len(coords[0]) > 2 and validate_polygon(coords, width, height):
                cv2.fillPoly(mask, coords, color=1)
                objects_processed += 1
            else:
                print(f"Poligono {leaf.tag} ignorado - invalido ou fora da imagem {filename}")
        except Exception as e:
            print(f"Erro ao processar {leaf.tag} em {filename}: {e}")

    # Quadrado (classe 2)
    square_elements = [child for child in object_element if child.tag == "square"]
    for square in square_elements:
        try:
            coords = extract_polygon_coords(square, width, height)
            if len(coords[0]) > 2 and validate_polygon(coords, width, height):
                cv2.fillPoly(mask, coords, color=2)
                objects_processed += 1
            else:
                print(f"Quadrado ignorado - invalido ou fora da imagem {filename}")
        except Exception as e:
            print(f"Erro ao processar quadrado em {filename}: {e}")

    # Salva a máscara
    out_name = base_name + "_label.png"
    out_path = os.path.join(output_mask_dir, out_name)
    cv2.imwrite(out_path, mask)

    print(f"OK {filename} -> {out_name} ({objects_processed} objetos)")

    if 2 not in np.unique(mask):
        print(f"AVISO: Classe 2 (quadrado) ausente na máscara de {out_name}")

    return True


# Processa todos os arquivos XML
xml_files = [f for f in os.listdir(xml_dir) if f.endswith(".xml")]
print(f"Encontrados {len(xml_files)} arquivos XML")

successful_conversions = 0
failed_conversions = 0

for xml_filename in xml_files:
    xml_path = os.path.join(xml_dir, xml_filename)
    if process_xml_to_mask(xml_path, image_dir, output_mask_dir):
        successful_conversions += 1
    else:
        failed_conversions += 1

print(f"\n=== RESUMO DA CONVERSAO ===")
print(f"OK Sucessos: {successful_conversions}")
print(f"X Falhas: {failed_conversions}")
print(f"Pasta Mascaras salvas em: {output_mask_dir}")

# Verifica as classes presentes nas máscaras geradas
print(f"\n=== VERIFICACAO DAS CLASSES ===")
label_files = [f for f in os.listdir(output_mask_dir) if f.endswith("_label.png")]
if label_files:
    sample_mask = cv2.imread(os.path.join(output_mask_dir, label_files[0]), cv2.IMREAD_GRAYSCALE)
    unique_values = np.unique(sample_mask)
    print(f"Classes encontradas na primeira mascara: {unique_values}")
    print("Mapeamento de classes:")
    print("  0 = Fundo")
    print("  1 = Folha")
    print("  2 = Quadrado/Retangulo")
else:
    print("Nenhuma mascara foi gerada!")
