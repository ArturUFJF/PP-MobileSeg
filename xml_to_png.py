import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

# === CAMINHOS ===
xml_dir = "data/dimensionamento_foliar2/xml"
image_dir = "data/dimensionamento_foliar2/images"
output_mask_dir = "data/dimensionamento_foliar2/annotations"

os.makedirs(output_mask_dir, exist_ok=True)

print("=== INICIANDO CONVERSAO (NOVA ESTRUTURA XML) ===")

def extract_polygon_coords_onevision(polygon_node, scale_factor):
    """
    Extrai coordenadas x1, y1, x2, y2... da tag <polygon>.
    Para esse formato OneVision, as coordenadas parecem ser normalizadas pela ALTURA.
    scale_factor deve ser a altura da imagem.
    """
    coords = []
    i = 1
    while True:
        x_tag = polygon_node.find(f'x{i}')
        y_tag = polygon_node.find(f'y{i}')
        
        if x_tag is None or y_tag is None:
            break
            
        try:
            # Multiplica AMBOS (X e Y) pela altura da imagem (scale_factor)
            x_val = float(x_tag.text) * scale_factor
            y_val = float(y_tag.text) * scale_factor
            coords.append((int(x_val), int(y_val)))
        except (ValueError, TypeError):
            pass
        
        i += 1

    return np.array([coords], dtype=np.int32)

def process_xml_to_mask(xml_path, image_dir, output_mask_dir):
    xml_filename = os.path.basename(xml_path)

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[ERRO] Falha ao ler XML {xml_filename}: {e}")
        return False

    # 1. Obter nome da imagem
    image_name_node = root.find("image-name")
    if image_name_node is None or not image_name_node.text:
        print(f"[PULAR] Tag <image-name> não encontrada em {xml_filename}")
        return False
    
    image_filename = image_name_node.text.strip()
    
    # 2. Carregar a imagem para garantir as dimensões corretas
    image_path = os.path.join(image_dir, image_filename)
    if not os.path.exists(image_path):
        # Tenta extensões alternativas caso o XML diga jpg mas seja png
        base, _ = os.path.splitext(image_filename)
        for ext in ['.jpg', '.png', '.jpeg']:
            if os.path.exists(os.path.join(image_dir, base + ext)):
                image_path = os.path.join(image_dir, base + ext)
                break
        
        if not os.path.exists(image_path):
            print(f"[ERRO] Imagem não encontrada: {image_filename}")
            return False

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERRO] Não foi possível abrir a imagem: {image_path}")
        return False

    height, width = img.shape[:2]
    
    # Cria máscara preta
    mask = np.zeros((height, width), dtype=np.uint8)

    # 3. Processar objetos
    # A estrutura agora é <objects> -> <leaf> ou <pattern>
    objects_node = root.find("objects")
    if objects_node is None:
        print(f"[AVISO] Tag <objects> não encontrada em {xml_filename}")
        return False

    count_leaves = 0
    count_patterns = 0

    # Itera sobre todos os filhos de <objects>
    for obj in objects_node:
        tag = obj.tag.lower()
        polygon_node = obj.find("polygon")
        
        if polygon_node is None:
            continue

        # Lógica de escala: Usamos a ALTURA para normalizar X e Y
        coords = extract_polygon_coords_onevision(polygon_node, scale_factor=height)
        
        if coords.size > 0:
            if tag == "leaf":
                # Classe 1: Folha
                cv2.fillPoly(mask, coords, color=1)
                count_leaves += 1
            elif tag == "pattern":
                # Classe 2: Padrão (antigo quadrado)
                cv2.fillPoly(mask, coords, color=2)
                count_patterns += 1

    # 4. Salvar
    out_name = os.path.splitext(os.path.basename(image_path))[0] + ".png"
    out_path = os.path.join(output_mask_dir, out_name)
    
    cv2.imwrite(out_path, mask)
    print(f"Gerado: {out_name} | Folhas: {count_leaves}, Padrões: {count_patterns}")
    return True

# === EXECUÇÃO ===
xml_files = [f for f in os.listdir(xml_dir) if f.endswith(".xml")]
print(f"Encontrados {len(xml_files)} arquivos XML.")

success = 0
for xf in xml_files:
    if process_xml_to_mask(os.path.join(xml_dir, xf), image_dir, output_mask_dir):
        success += 1

print(f"\nConcluído! {success}/{len(xml_files)} máscaras geradas.")

# Verificação rápida da última máscara
files = sorted([f for f in os.listdir(output_mask_dir) if f.endswith(".png")])
if files:
    last_mask = cv2.imread(os.path.join(output_mask_dir, files[-1]), cv2.IMREAD_GRAYSCALE)
    print(f"\nClasses na última máscara ({files[-1]}): {np.unique(last_mask)}")
    print("(Deve conter: 0=Fundo, 1=Folha, 2=Padrão/Quadrado)")