import os
import re

def fix_xml_file(file_path):
    """
    Fix malformed XML tags like <y1>1040<y1/> to <y1>1040</y1>
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Pattern to match malformed tags like <y1>1040<y1/> or <x1>327<x1/>
        pattern = r'<([xy]\d+)>(\d+)<\1/>'
        replacement = r'<\1>\2</\1>'
        
        # Apply the fix
        fixed_content = re.sub(pattern, replacement, content)
        
        # Check if any changes were made
        if fixed_content != content:
            # Create backup
            backup_path = file_path + '.backup'
            if not os.path.exists(backup_path):
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            # Write fixed content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            
            return True
        return False
        
    except Exception as e:
        print(f"Erro ao processar {file_path}: {e}")
        return False

def main():
    xml_dir = "data/dimensionamento_foliar/masks"
    
    if not os.path.exists(xml_dir):
        print(f"Diretório não encontrado: {xml_dir}")
        return
    
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    fixed_count = 0
    
    print(f"Processando {len(xml_files)} arquivos XML...")
    
    for filename in xml_files:
        file_path = os.path.join(xml_dir, filename)
        if fix_xml_file(file_path):
            fixed_count += 1
            print(f"Corrigido: {filename}")
    
    print(f"\nConcluído! {fixed_count} arquivos foram corrigidos.")
    print("Backups dos arquivos originais foram criados com extensão .backup")

if __name__ == "__main__":
    main()
