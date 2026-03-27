# Script para aumentar a resolução (upscale) de predições de 640x640 para o tamanho original das imagens
# Lê predições em baixa resolução e as amplia usando interpolação para corresponder ao tamanho original

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# Função para analisar argumentos de linha de comando
def parse_args():
    parser = argparse.ArgumentParser(
        description="Aumentar resolução de predições de 640x640 para tamanho original da imagem."
    )
    # Diretório com predições em baixa resolução (ex: pseudo_color_prediction/)
    parser.add_argument(
        "--pred_dir",
        type=str,
        required=True,
        help="Diretório com predições (ex: pseudo_color_prediction/).",
    )
    # Diretório com imagens originais para determinar o tamanho alvo
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Diretório com imagens originais.",
    )
    # Diretório de saída para as predições ampliadas
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Diretório para salvar predições aumentadas.",
    )
    # Método de interpolação para ampliar: nearest, bilinear ou bicubic
    parser.add_argument(
        "--interp",
        type=str,
        default="nearest",
        choices=["nearest", "bilinear", "bicubic"],
        help="Método de interpolação para upscaling (padrão: nearest).",
    )
    # Opção para gerar também imagens de sobreposição (overlay)
    parser.add_argument(
        "--save_overlay",
        action="store_true",
        help="Também salvar imagens de sobreposição (máscara predita sobreposta à imagem original).",
    )
    # Diretório customizado para salvar overlays
    parser.add_argument(
        "--overlay_dir",
        type=str,
        default=None,
        help="Diretório para salvar imagens de sobreposição. Se omitido, usa <output_dir>/added_prediction.",
    )
    # Peso de transparência da máscara em overlays (0.0 = imagem pura, 1.0 = máscara pura)
    parser.add_argument(
        "--overlay_weight",
        type=float,
        default=0.6,
        help="Peso de transparência da máscara no overlay (0.0-1.0). Padrão: 0.6.",
    )
    # Paleta de cores customizada para colorização (lista flat de valores RGB)
    parser.add_argument(
        "--custom_color",
        nargs="+",
        type=int,
        default=None,
        help="Paleta de cores customizada como lista RGB flat, ex: --custom_color 0 0 0 255 0 0 0 0 255",
    )
    return parser.parse_args()


# Função para obter as dimensões (altura, largura) da imagem original
def get_original_size(image_path: Path) -> tuple:
    """Get (H, W) of original image."""
    # Abre a imagem usando PIL
    img = Image.open(image_path)
    # PIL retorna (largura, altura), então inverte para (altura, largura)
    return img.size[::-1]


# Função para carregar uma predição e aumentar sua resolução para o tamanho alvo
def upscale_mask(pred_path: Path, target_size: tuple, interp: str) -> np.ndarray:
    """Load prediction and upscale to target size."""
    # Abre a imagem PNG da predição
    pred = Image.open(pred_path)
    # Se a imagem for em modo paleta (indexed), extrai a paleta para preservá-la
    palette = pred.getpalette() if pred.mode == "P" else None
    # Converte a imagem em array numpy
    pred_arr = np.array(pred)

    # Trata imagens RGB: converte para escala de cinza pegando o primeiro canal
    # (em imagens de classe indexados, todos os canais têm o mesmo valor)
    if pred_arr.ndim == 3:
        pred_arr = pred_arr[:, :, 0]

    # Mapa de métodos de interpolação do OpenCV
    # nearest: mais rápido, preserva valores exatos
    # bilinear: interpolação linear, resulta em bordas suavizadas
    # bicubic: interpolação cúbica, qualidade mais alta mas mais lenta
    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
    }
    # Redimensiona a imagem usando OpenCV para o tamanho alvo
    # target_size é (altura, largura), mas cv2.resize espera (largura, altura)
    upscaled = cv2.resize(pred_arr, (target_size[1], target_size[0]), interpolation=interp_map[interp])
    return upscaled, palette


# Função para salvar uma máscara em arquivo PNG
def save_mask(mask: np.ndarray, output_path: Path, palette=None):
    """Save mask as image."""
    # Cria diretórios necessários se não existirem
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Se existe uma paleta, salva como imagem indexada (modo "P") com paleta
    if palette is not None:
        # Cria imagem em modo paleta
        img = Image.fromarray(mask, mode="P")
        # Aplica a paleta
        img.putpalette(palette)
        # Salva em arquivo
        img.save(output_path)
    else:
        # Caso contrário, salva como imagem de escala de cinza simples
        Image.fromarray(mask).save(output_path)


# Função para construir uma paleta de cores para colorização das máscaras
def build_palette(custom_color=None) -> np.ndarray:
    """Build palette as [N, 3] RGB uint8."""
    # Se uma paleta customizada foi fornecida, a converte em formato apropriado
    if custom_color is not None:
        # Valida que o comprimento é múltiplo de 3 (3 componentes RGB por cor)
        if len(custom_color) % 3 != 0:
            raise ValueError("--custom_color deve ter comprimento múltiplo de 3.")
        # Converte para array e reshape para (num_cores, 3)
        palette = np.array(custom_color, dtype=np.uint8).reshape(-1, 3)
        return palette

    # Paleta padrão de 3 classes: fundo (preto), classe-1 (vermelho), classe-2 (azul)
    return np.array(
        [
            [0, 0, 0],      # Classe 0: preto (fundo)
            [255, 0, 0],    # Classe 1: vermelho
            [0, 0, 255],    # Classe 2: azul
        ],
        dtype=np.uint8,
    )


# Função para converter uma máscara de índices de classe em imagem RGB colorizada
def colorize_mask(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Map class indices to RGB colors."""
    # Encontra o índice de classe máximo na máscara
    max_cls = int(mask.max()) if mask.size else 0
    # Se a classe máxima excede a paleta, expande a paleta com cores pretas
    if max_cls >= len(palette):
        extra = max_cls + 1 - len(palette)
        palette = np.vstack([palette, np.zeros((extra, 3), dtype=np.uint8)])
    # Mapeia cada índice de classe para sua cor RGB usando a paleta
    return palette[mask]


# Função para salvar uma imagem de sobreposição (blend da imagem original com máscara colorizada)
def save_overlay_image(
    image_path: Path,
    mask: np.ndarray,
    overlay_path: Path,
    palette: np.ndarray,
    overlay_weight: float,
):
    """Save overlay image by blending original image and colorized mask."""
    # Cria diretório de saída se necessário
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    # Abre a imagem original e converte para RGB (caso necessário)
    image = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    # Coloriza a máscara predita
    color_mask = colorize_mask(mask, palette)
    # Realiza blend (mistura) entre imagem original e máscara usando cv2.addWeighted
    # Fórmula: resultado = imagem * (1 - overlay_weight) + color_mask * overlay_weight
    blended = cv2.addWeighted(image, 1.0 - overlay_weight, color_mask, overlay_weight, 0.0)
    # Salva a imagem resultante
    Image.fromarray(blended).save(overlay_path)


# Função principal que orquestra todo o processo de aumento de resolução
def main():
    # Analisa argumentos de linha de comando
    args = parse_args()

    # Converte strings em objetos Path para manipulação de caminhos
    pred_dir = Path(args.pred_dir)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    # Define diretório de overlay: usa diretório customizado ou cria um padrão
    overlay_dir = Path(args.overlay_dir) if args.overlay_dir else output_dir / "added_prediction"

    # Valida que o weight de overlay está no intervalo [0.0, 1.0]
    if not (0.0 <= args.overlay_weight <= 1.0):
        raise ValueError("--overlay_weight deve estar no intervalo [0.0, 1.0].")

    # Constrói a paleta de cores para colorização
    palette = build_palette(args.custom_color)

    # Valida existência dos diretórios de entrada
    if not pred_dir.exists():
        raise FileNotFoundError(f"Diretório de predições não encontrado: {pred_dir}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Diretório de imagens não encontrado: {image_dir}")

    # Coleta todos os arquivos PNG de predição
    pred_files = sorted([p for p in pred_dir.rglob("*.png")])
    if not pred_files:
        raise RuntimeError(f"Nenhum arquivo PNG encontrado em {pred_dir}")

    # Exibe informações gerais sobre a execução
    print(f"Encontrados {len(pred_files)} arquivos para upscaling.")
    print(f"Método de interpolação: {args.interp}")

    # Contadores para estatísticas de execução
    processed = 0        # Predições processadas com sucesso
    skipped = 0          # Predições puladas (sem imagem correspondente, etc.)
    overlays_saved = 0   # Imagens de overlay salvas
    total_upscaled_pixels = 0  # Total de pixels processados

    # Loop principal: processa cada arquivo de predição
    for idx, pred_file in enumerate(pred_files, start=1):
        # Extrai o nome do arquivo sem extensão
        stem = pred_file.stem

        # Tenta encontrar a imagem original com diferentes extensões
        img_file = None
        for ext in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]:
            img_candidate = image_dir / f"{stem}{ext}"
            if img_candidate.exists():
                img_file = img_candidate
                break

        # Se não encontrou imagem correspondente, pula este arquivo
        if img_file is None:
            print(f"PULA: {pred_file.name} (nenhuma imagem correspondente)")
            skipped += 1
            continue

        try:
            # Obtém o tamanho original da imagem
            orig_size = get_original_size(img_file)
            # Carrega e upscala a predição para o tamanho original
            upscaled, pred_palette = upscale_mask(pred_file, orig_size, args.interp)

            # Salva a predição upscalada
            output_file = output_dir / pred_file.name
            save_mask(upscaled, output_file, pred_palette)

            # Opcionalmente salva overlay (predição sobreposta à imagem original)
            if args.save_overlay:
                overlay_file = overlay_dir / pred_file.name
                save_overlay_image(
                    image_path=img_file,
                    mask=upscaled,
                    overlay_path=overlay_file,
                    palette=palette,
                    overlay_weight=args.overlay_weight,
                )
                overlays_saved += 1

            # Acumula estatísticas de pixels processados
            pred_size = upscaled.shape
            total_upscaled_pixels += pred_size[0] * pred_size[1]
            processed += 1

            # Exibe progresso a cada 50 arquivos ou no final
            if idx % 50 == 0 or idx == len(pred_files):
                print(f"Processados {idx}/{len(pred_files)} arquivos")
        except Exception as e:
            # Em caso de erro, registra e continua com próximo arquivo
            print(f"ERRO {pred_file.name}: {e}")
            skipped += 1

    # Exibe resumo final da execução
    print(f"\nConcluído.")
    print(f"  Processados: {processed}")
    print(f"  Pulados: {skipped}")
    if args.save_overlay:
        print(f"  Overlays salvos: {overlays_saved}")
        print(f"  Diretório de overlays: {overlay_dir}")
    print(f"  Total de pixels upscalados: {total_upscaled_pixels:,}")
    print(f"  Diretório de saída: {output_dir}")


# Point de entrada do script: executa a função main() quando o script é rodado diretamente
if __name__ == "__main__":
    main()
