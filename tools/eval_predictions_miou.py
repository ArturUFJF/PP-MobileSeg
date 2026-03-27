# Script para avaliar máscaras de segmentação preditas e calcular métricas de desempenho (mIoU, Acurácia, etc.)
# Compara predições com imagens de verdade absoluta (ground-truth) pixel por pixel

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


# Função para analisar argumentos de linha de comando
def parse_args():
    parser = argparse.ArgumentParser(
        description="Avaliar máscaras de segmentação e gerar relatório de mIoU."
    )
    # Diretório contendo as máscaras preditas pelo modelo (em formato PNG)
    parser.add_argument(
        "--pred_dir",
        type=str,
        required=True,
        help="Diretório com máscaras preditas (.png).",
    )
    # Diretório contendo as máscaras de verdade absoluta (ground-truth) para validação
    parser.add_argument(
        "--gt_dir",
        type=str,
        required=True,
        help="Diretório com máscaras de verdade absoluta (.png).",
    )
    # Arquivo opcional que define um subconjunto de imagens para avaliação
    # Formato: cada linha contém "caminho_imagem caminho_label"
    parser.add_argument(
        "--split_file",
        type=str,
        default=None,
        help="Arquivo txt opcional (caminho_imagem caminho_label) para definir subset.",
    )
    # Número total de classes de segmentação (ex: fundo, classe1, classe2 = 3 classes)
    parser.add_argument(
        "--num_classes",
        type=int,
        default=3,
        help="Número de classes de segmentação.",
    )
    # Índice de classe a ser ignorado na avaliação (geralmente usado para pixels inválidos)
    parser.add_argument(
        "--ignore_index",
        type=int,
        default=255,
        help="Índice de classe a ignorar nas máscaras de verdade absoluta.",
    )
    return parser.parse_args()


# Função para ler uma máscara de índice de classe a partir de um arquivo PNG
def read_mask(path: Path) -> np.ndarray:
    # Abre a imagem PNG e converte para array numpy
    mask = np.array(Image.open(path))
    # Se a imagem tiver 3 canais (RGB), pega apenas o primeiro canal
    # Pois em imagens indexadas, todos os canais têm o mesmo valor
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    # Converte para tipo int64 para cálculos posteriores
    return mask.astype(np.int64)


# Função que calcula a matriz de confusão (histograma) entre labels verdadeiros e preditos
# Esta matriz é usada para calcular métricas como IoU e acurácia
def fast_hist(gt: np.ndarray, pred: np.ndarray, num_classes: int, ignore_index: int):
    # Cria máscara de pixels válidos:
    # - Pixels que não são índice de ignore
    # - Pixels com valor >= 0
    # - Pixels com valor < num_classes
    valid = (gt != ignore_index) & (gt >= 0) & (gt < num_classes)
    
    # Filtra apenas os pixels válidos das máscaras ground-truth e predita
    gt = gt[valid]
    pred = pred[valid]
    
    # Se não há pixels válidos, retorna matriz de confusão vazia
    if gt.size == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64), 0
    
    # Verifica se há índices de classe inválidos nas predições
    invalid_pred = (pred < 0) | (pred >= num_classes)
    if np.any(invalid_pred):
        # Se houver, lança erro informando quais valores inválidos foram encontrados
        invalid_values = np.unique(pred[invalid_pred])
        raise ValueError(
            f"Máscara predita contém índices de classe inválidos: {invalid_values.tolist()} (intervalo válido: 0..{num_classes - 1})"
        )
    
    # Calcula a matriz de confusão usando bincount:
    # Converte cada par (gt, pred) em um índice único: gt * num_classes + pred
    # Conta quantas vezes cada combinação ocorre
    hist = np.bincount(num_classes * gt + pred, minlength=num_classes**2)
    
    # Reshape em matriz quadrada (num_classes x num_classes) e retorna contagem de pixels válidos
    return hist.reshape(num_classes, num_classes), valid.sum()


# Função para coletar arquivos de ground-truth para avaliação
# Pode usar todos os arquivos PNG do diretório ou apenas um subconjunto definido por arquivo de divisão
def collect_gt_files(gt_dir: Path, split_file: Path | None):
    # Se não há arquivo de divisão especificado, usa todos os PNGs do diretório
    if split_file is None:
        return sorted(gt_dir.rglob("*.png"))

    # Caso contrário, lê o arquivo de divisão e extrai os caminhos de ground-truth
    gt_files = []
    for line in split_file.read_text(encoding="utf-8").splitlines():
        # Remove espaços em branco da linha
        line = line.strip()
        # Ignora linhas vazias
        if not line:
            continue
        # Divide a linha em partes (esperado: "caminho_imagem caminho_label")
        parts = line.split()
        # Verifica se a linha tem pelo menos 2 partes
        if len(parts) < 2:
            continue
        # Extrai o caminho do label (segunda parte) e normaliza barras invertidas
        rel_gt = parts[1].replace("\\", "/")
        # Constrói caminho absoluto e adiciona à lista
        gt_files.append(gt_dir.parent / rel_gt)
    return gt_files


# Função para criar um índice (mapa) de arquivos de predição
# Facilita a busca rápida de um arquivo predito pelo nome da imagem
def build_pred_index(pred_dir: Path):
    pred_map = {}  # Dicionário: nome_arquivo -> caminho_completo
    # Percorre recursivamente todos os arquivos PNG no diretório
    for p in pred_dir.rglob("*.png"):
        # Verifica se já existe um arquivo com o mesmo nome (ambiguidade)
        if p.name in pred_map:
            raise ValueError(
                f"Nome de arquivo de predição duplicado encontrado: {p.name}\n"
                f"  Primeiro: {pred_map[p.name]}\n"
                f"  Segundo: {p}\n"
                "Use nomes de arquivo únicos ou avalie uma pasta de cada vez."
            )
        # Armazena o caminho completo indexado pelo nome do arquivo
        pred_map[p.name] = p
    return pred_map


# Função principal que orquestra todo o processo de avaliação
def main():
    # Analisa argumentos de linha de comando
    args = parse_args()

    # Converte strings de argumentos em objetos Path para melhor manipulação de caminhos
    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    split_file = Path(args.split_file) if args.split_file else None

    # Valida se os diretórios existem
    if not pred_dir.exists():
        raise FileNotFoundError(f"Diretório de predições não encontrado: {pred_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"Diretório de ground-truth não encontrado: {gt_dir}")
    if split_file and not split_file.exists():
        raise FileNotFoundError(f"Arquivo de divisão não encontrado: {split_file}")

    # Coleta todos os arquivos de ground-truth a serem avaliados
    gt_files = collect_gt_files(gt_dir, split_file)
    if not gt_files:
        raise RuntimeError("Nenhum arquivo de ground-truth encontrado para avaliação.")

    # Cria índice rápido de arquivos de predição para busca eficiente
    pred_index = build_pred_index(pred_dir)

    # Inicializa matriz de confusão acumulada (começando com zeros)
    confusion = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    # Contadores para estatísticas
    total_valid_pixels = 0  # Total de pixels válidos avaliados
    evaluated = 0  # Número de pares imagem-predição avaliados com sucesso
    missing = 0  # Número de arquivos faltando ou com problemas

    # Loop principal: processa cada arquivo de ground-truth
    for gt_file in gt_files:
        # Verifica se o arquivo de ground-truth existe
        if not gt_file.exists():
            missing += 1
            print(f"PULA GT ausente: {gt_file}")
            continue

        # Busca o arquivo de predição correspondente usando o índice
        pred_file = pred_index.get(gt_file.name)
        if pred_file is None:
            missing += 1
            print(f"PULA PRED ausente: {gt_file.name}")
            continue

        # Carrega as máscaras de ground-truth e predita
        gt = read_mask(gt_file)
        pred = read_mask(pred_file)

        # Valida se as dimensões das máscaras correspondem
        if gt.shape != pred.shape:
            print(f"PULA TAMANHO incorreto: {gt_file.name} gt={gt.shape} pred={pred.shape}")
            missing += 1
            continue

        # Calcula a matriz de confusão para este par de máscaras
        hist, valid_count = fast_hist(gt, pred, args.num_classes, args.ignore_index)
        # Acumula na matriz de confusão total
        confusion += hist
        total_valid_pixels += int(valid_count)
        evaluated += 1

    # Verifica se pelo menos um arquivo foi avaliado com sucesso
    if evaluated == 0:
        raise RuntimeError("Nenhum arquivo foi avaliado. Verifique caminhos e nomes de arquivo.")

    # Extrai componentes da matriz de confusão para cálculo de métricas
    
    # Verdadeiros positivos (TP): diagonal da matriz de confusão
    # Valores onde ground-truth e predição coincidem
    tp = np.diag(confusion).astype(np.float64)
    
    # Soma de cada linha: total de pixels daquela classe no ground-truth
    gt_sum = confusion.sum(axis=1).astype(np.float64)
    
    # Soma de cada coluna: total de pixels que o modelo predisse como aquela classe
    pred_sum = confusion.sum(axis=0).astype(np.float64)

    # Calcula IoU (Intersection over Union) para cada classe
    # Fórmula: tp / (gt_sum + pred_sum - tp)
    # O máximo com 1.0 evita divisão por zero
    iou = tp / np.maximum(gt_sum + pred_sum - tp, 1.0)
    
    # Calcula acurácia por classe (Recall)
    # Fórmula: tp / total de pixels daquela classe
    class_acc = tp / np.maximum(gt_sum, 1.0)
    
    # Calcula métricas globais (média ignorando NaN)
    miou = float(np.nanmean(iou))  # Média de IoU entre todas as classes
    macc = float(np.nanmean(class_acc))  # Média de acurácia entre todas as classes
    pixacc = float(tp.sum() / np.maximum(confusion.sum(), 1.0))  # Acurácia geral de pixels

    # Imprime resumo da avaliação
    print("\n=== Resumo da Avaliação ===")
    print(f"Arquivos avaliados: {evaluated}")
    print(f"Ausentes/Pulados: {missing}")
    print(f"Pixels válidos: {total_valid_pixels:,}")
    print(f"Acurácia de Pixel: {pixacc:.6f}")
    print(f"Acurácia Média: {macc:.6f}")
    print(f"mIoU: {miou:.6f}")

    # Imprime métricas detalhadas para cada classe
    for c in range(args.num_classes):
        print(f"Classe {c} IoU: {iou[c]:.6f} | Acurácia: {class_acc[c]:.6f}")


# Point de entrada do script: executa a função main() quando o script é rodado diretamente
if __name__ == "__main__":
    main()
