import os
import xml.etree.ElementTree as ET
import re

def normalize_filename_content(filename):
    """
    Normaliza o conteúdo do filename:
    - Converte para minúsculas
    - Remove espaços 
    - Troca .png por .jpg
    """
    # Converte para minúsculas
    filename = filename.lower()
    
    # Remove espaços
    filename = filename.replace(' ', '')
    
    # Troca .png por .jpg
    filename = filename.replace('.png', '.jpg')
    
    return filename

def fix_xml_file(xml_path):
    """Corrige o elemento <filename> em um arquivo XML"""
    try:
        # Faz o parse do XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Encontra o elemento filename
        filename_element = root.find("filename")
        if filename_element is None:
            print(f"AVISO: Elemento 'filename' não encontrado em {xml_path}")
            return False
        
        # Pega o valor original
        original_filename = filename_element.text
        if original_filename is None:
            print(f"AVISO: Elemento 'filename' vazio em {xml_path}")
            return False
        
        # Normaliza o filename
        new_filename = normalize_filename_content(original_filename)
        
        # Verifica se houve mudança
        if original_filename != new_filename:
            # Atualiza o elemento
            filename_element.text = new_filename
            
            # Salva o arquivo
            tree.write(xml_path, encoding='utf-8', xml_declaration=True)
            
            print(f"✓ {os.path.basename(xml_path)}: '{original_filename}' -> '{new_filename}'")
            return True
        else:
            print(f"- {os.path.basename(xml_path)}: Nenhuma mudança necessária")
            return False
            
    except ET.ParseError as e:
        print(f"✗ Erro de parse em {xml_path}: {e}")
        return False
    except Exception as e:
        print(f"✗ Erro inesperado em {xml_path}: {e}")
        return False

def main():
    # Diretório com os arquivos XML
    xml_dir = "data/dimensionamento_foliar/masks"
    
    if not os.path.exists(xml_dir):
        print(f"Diretório não encontrado: {xml_dir}")
        return
    
    # Lista todos os arquivos XML
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith(".xml") and not f.endswith(".backup")]
    
    print(f"Encontrados {len(xml_files)} arquivos XML para processar")
    print("=" * 60)
    
    # Contadores
    modified_count = 0
    unchanged_count = 0
    error_count = 0
    
    # Processa cada arquivo
    for xml_filename in sorted(xml_files):
        xml_path = os.path.join(xml_dir, xml_filename)
        
        result = fix_xml_file(xml_path)
        if result is True:
            modified_count += 1
        elif result is False:
            unchanged_count += 1
        else:
            error_count += 1
    
    # Resumo
    print("=" * 60)
    print(f"RESUMO:")
    print(f"✓ Arquivos modificados: {modified_count}")
    print(f"- Arquivos sem mudança: {unchanged_count}")
    print(f"✗ Erros: {error_count}")
    print(f"📁 Total processado: {len(xml_files)}")

if __name__ == "__main__":
    main()
