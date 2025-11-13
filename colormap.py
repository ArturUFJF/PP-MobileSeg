import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import glob
import argparse
import os

# --- Argumentos de linha de comando ---
parser = argparse.ArgumentParser(description='Gera colormaps otimizados.')
parser.add_argument('--exp_path', type=str, required=True, help='Path da experimentação')
args = parser.parse_args()

raw_path = glob.glob(f'SegFormer/work_dirs/{args.exp_path}/outputs_val_area/area_results/*_area.raw')
image_shape = (512, 512)  # altura e largura da imagem

# === Criar colormap custom uma única vez ===
base_cmap = plt.cm.jet
cmap_array = base_cmap(np.linspace(0, 1, 256))
cmap_array[0] = [0, 0, 0, 1]  # fundo = preto
custom_cmap = LinearSegmentedColormap.from_list('cmap', cmap_array, 256)

# === Preparar figura e axis uma única vez ===
fig, ax = plt.subplots()
im = ax.imshow(np.zeros(image_shape), cmap=custom_cmap, vmin=0, vmax=1)
ax.set_axis_off()
fig.colorbar(im, ax=ax)

# === Função de salvamento reutilizando figura ===
def save_colormap(data, out_path, vmin=0, vmax=None):
    im.set_data(data)
    im.set_clim(vmin, vmax)
    fig.canvas.draw_idle()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')

# === Loop pelos arquivos ===
for fname in raw_path:
    # carregar dados
    image1_data = np.fromfile(fname, dtype=np.float32).reshape(image_shape) / 1000.0
    fname2 = fname.replace('_area.', '_leaf.')
    image2_data = np.fromfile(fname2, dtype=np.float32).reshape(image_shape) / 1000.0
    fname3 = fname.replace('_area.', '_marker.')
    image3_data = np.fromfile(fname3, dtype=np.float32).reshape(image_shape) / 1000.0

    # usar valor máximo consistente (aqui: da área, igual ao seu código original)
    max_value = np.max(image1_data)

    # salvar
    save_colormap(image1_data, fname.replace('.raw', '_colormap.png'), vmin=0, vmax=max_value)
    save_colormap(image2_data, fname2.replace('.raw', '_colormap.png'), vmin=0, vmax=max_value)
    save_colormap(image3_data, fname3.replace('.raw', '_colormap.png'), vmin=0, vmax=max_value)

plt.close(fig)
