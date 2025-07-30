import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

# Caminhos
xml_dir = "data/dimensionamento_foliar/masks"  # onde estão os arquivos .xml
image_dir = "data/dimensionamento_foliar/images"  # onde estão as imagens .jpg ou .png
output_mask_dir = "data/dimensionamento_foliar/labels"  # onde as máscaras serão salvas

os.makedirs(output_mask_dir, exist_ok=True)

print(f"Processando XMLs do diretório: {xml_dir}")
print(f"Buscando imagens no diretório: {image_dir}")
print(f"Salvando máscaras em: {output_mask_dir}")

xml_files = [f for f in os.listdir(xml_dir) if f.endswith(".xml")]
print(f"Encontrados {len(xml_files)} arquivos XML")

def extract_polygon_coords(obj):
    coords = []
    for i in range(1, 1000):
        x_tag = obj.find(f'x{i}')
        y_tag = obj.find(f'y{i}')
        if x_tag is None or y_tag is None:
            break
        
        try:
            x_val = int(x_tag.text) if x_tag.text else 0
            y_val = int(y_tag.text) if y_tag.text else 0
            coords.append((x_val, y_val))
        except (ValueError, TypeError) as e:
            print(f"Erro ao converter coordenadas x{i}, y{i}: {e}")
            continue
    
    return np.array([coords], dtype=np.int32)

for filename in os.listdir(xml_dir):
    if not filename.endswith(".xml"):
        continue

    xml_path = os.path.join(xml_dir, filename)
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Erro ao fazer parse do XML {filename}: {e}")
        continue
    except Exception as e:
        print(f"Erro inesperado ao processar {filename}: {e}")
        continue

    # Encontra o nome da imagem associada
    filename_element = root.find("filename")
    if filename_element is None or filename_element.text is None:
        print(f"Elemento 'filename' não encontrado em {filename}")
        continue
    
    # Processa o nome da imagem: remove _mask.png, converte .png para .jpg, e deixa minúsculo
    original_name = filename_element.text
    image_name = original_name.replace("_mask.png", "").replace(".png", "").replace(".jpg", "")
    image_name = image_name.lower() + ".jpg"
    
    # Tenta encontrar a imagem com diferentes extensões e casos
    image_path = None
    possible_extensions = [".jpg", ".png", ".JPG", ".PNG"]
    possible_names = [
        image_name,  # nome em minúsculo com .jpg
        original_name.replace("_mask.png", ".jpg"),  # nome original com .jpg
        original_name.replace("_mask.png", ".png"),  # nome original com .png
        original_name.replace("_mask.png", ".JPG"),  # nome original com .JPG
        original_name.replace("_mask.png", ".PNG")   # nome original com .PNG
    ]
    
    for test_name in possible_names:
        test_path = os.path.join(image_dir, test_name)
        if os.path.exists(test_path):
            image_path = test_path
            image_name = test_name
            break
    
    if image_path is None:
        print(f"Imagem não encontrada para {filename}. Tentativas: {possible_names}")
        continue

    # Detecta tamanho da imagem
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro ao abrir: {image_path}")
        continue
    height, width = img.shape[:2]

    # Cria máscara do mesmo tamanho
    mask = np.zeros((height, width), dtype=np.uint8)

    # Busca pelo elemento 'object'
    object_element = root.find("object")
    if object_element is not None:
        # Processa todos os elementos filhos do object (leaf1, leaf2, etc.)
        leaf_elements = [child for child in object_element if child.tag.startswith('leaf')]
        
        for idx, leaf in enumerate(leaf_elements):
            try:
                coords = extract_polygon_coords(leaf)
                if len(coords[0]) > 2:  # Precisa de pelo menos 3 pontos para um polígono
                    cv2.fillPoly(mask, coords, color=1)  # Todas as folhas = classe 1
                else:
                    print(f"Polígono insuficiente em {filename}, {leaf.tag}")
            except Exception as e:
                print(f"Erro ao processar {leaf.tag} em {filename}: {e}")
                continue

    # Salva a máscara
    base_name = os.path.splitext(image_name)[0]  # Remove a extensão
    out_name = base_name + "_label.png"
    out_path = os.path.join(output_mask_dir, out_name)
    cv2.imwrite(out_path, mask)
    print(f"Salvo: {out_path}")
