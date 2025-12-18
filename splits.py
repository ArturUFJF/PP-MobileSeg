import pandas as pd
import os

pasta_split = 'data/lsidbeans/cv_1/split_1'
sufixos = ['.jpg', '.png', '_area.raw'] # Ex: imagem, anotação, máscara

# CONFIGURAÇÃO: Dicionário mapeando "Arquivo CSV" -> "Seus Prefixos"
# Isso elimina a necessidade de múltiplos IFs
mapa_arquivos = {
    'train.csv': ['images/', 'annotations/', 'area_masks/'],
    'val.csv':   ['images/',   'annotations/',   'area_masks/'],
    'test.csv':  ['images/',  'annotations/',  'area_masks/']
}

# Iteramos direto sobre o dicionário (muito mais rápido e limpo)
for nome_csv, prefixos in mapa_arquivos.items():
    
    caminho_csv = os.path.join(pasta_split, nome_csv)
    
    # Verifica se o arquivo existe antes de tentar ler
    if os.path.exists(caminho_csv):
        print(f"Processando: {nome_csv}...")
        
        # 1. Lê o CSV (assumindo que o nome do arquivo está na primeira coluna '0')
        # header=None garante que ele leia a primeira linha como dados, não cabeçalho
        df = pd.read_csv(caminho_csv, header=None)
        
        # Converter para string para garantir que não dê erro na concatenação
        coluna_nomes = df[0].astype(str)
        
        # 2. Cria a linha completa vetorizada (O Pandas faz isso instantaneamente para todas as linhas)
        # Lógica: Prefixo + NomeArquivo + Sufixo
        caminho_img  = prefixos[0] + coluna_nomes + sufixos[0]
        caminho_ann  = prefixos[1] + coluna_nomes + sufixos[1]
        caminho_mask = prefixos[2] + coluna_nomes + sufixos[2]
        
        # Junta tudo com espaço
        linhas_finais = caminho_img + " " + caminho_ann + " " + caminho_mask
        
        # 3. Salva o resultado (Ex: cria um arquivo train.txt na mesma pasta)
        nome_saida = nome_csv.replace('.csv', '.txt')
        caminho_saida = os.path.join(pasta_split, nome_saida)
        
        linhas_finais.to_csv(caminho_saida, index=False, header=False)
        print(f"-> Salvo em: {nome_saida}")
        
    else:
        print(f"Aviso: {nome_csv} não encontrado em {pasta_split}")