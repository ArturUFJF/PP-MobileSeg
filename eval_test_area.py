import os
import numpy as np
from PIL import Image
import csv
import argparse
import sys

# Permite importar area_dict
sys.path.append('SegFormer/data/lsidbeans/')
from area_dict import area_dict  # supondo que o dicionário se chame area_dict

# --- Argumentos de linha de comando ---
parser = argparse.ArgumentParser(description='Processa áreas de folhas e marcadores.')
parser.add_argument('--split_path', type=str, required=True, help='Path do split, ex: cv_2_split_5')
parser.add_argument('--exp_path', type=str, required=True, help='Path da experimentação, ex: cv_2_split_5/exp_20250918_104405')
args = parser.parse_args()

area_mask_folder = os.path.join('SegFormer/data/lsidbeans', args.split_path, 'area_masks/val')
segmentation_mask_folder = os.path.join('SegFormer/data/lsidbeans', args.split_path, 'annotations/val')

area_output_folder_path = os.path.join('SegFormer/work_dirs', args.exp_path, 'outputs_val_area/area_results')
csv_path = os.path.join('SegFormer/work_dirs', args.exp_path, 'outputs_val_area/test_area.csv')

# Remove o CSV se ele existir
if os.path.isfile(csv_path):
    os.remove(csv_path)

true_leaf = []
pred_leaf = []
true_marker = []
pred_marker = []

leaf_files = [f for f in os.listdir(area_output_folder_path) if f.endswith('_leaf.raw')]

for filename in leaf_files:
    leaf_number = filename.split('_')[1]

    seg_mask = np.array(Image.open(os.path.join(segmentation_mask_folder, filename.replace('_leaf.raw', '.png'))))
    area_mask = np.fromfile(os.path.join(area_mask_folder, filename.replace('_leaf', '_area')), dtype=np.float32).reshape((512, 512))
    leaf_pred = np.fromfile(os.path.join(area_output_folder_path, filename), dtype=np.float32).reshape(512, 512)

    leaf_seg_mask = (seg_mask == 1)
    leaf_region_area_mask = np.zeros((512, 512), dtype=np.float32)
    leaf_region_area_mask[leaf_seg_mask] = area_mask[leaf_seg_mask]
    leaf_area = area_dict[leaf_number]
    leaf_area2 = np.sum(leaf_region_area_mask)/1000

    pred_sum_leaf = np.sum(leaf_pred)/1000

    true_leaf.append(leaf_area)
    pred_leaf.append(pred_sum_leaf)

    marker_pred = np.fromfile(os.path.join(area_output_folder_path, filename.replace('_leaf', '_marker')), dtype=np.float32).reshape(512, 512)

    marker_seg_mask = (seg_mask == 2)
    marker_region_area_mask = np.zeros((512, 512), dtype=np.float32)
    marker_region_area_mask[marker_seg_mask] = area_mask[marker_seg_mask]
    marker_area = np.sum(marker_region_area_mask)/1000

    pred_sum_marker = np.sum(marker_pred)/1000

    true_marker.append(marker_area)
    pred_marker.append(pred_sum_marker)

    print('>>> LEAF ', leaf_number, '\nPred: ', pred_sum_leaf, '\nTrue: ', leaf_area2, ', ~',  leaf_area, '\nTrue - Pred = ', str(leaf_area2 - pred_sum_leaf))
    print('MARKER\nPred: ', pred_sum_marker, '\nTrue: ', marker_area, ', ~25.0', '\nTrue - Pred = ', str(marker_area - pred_sum_marker))

    with open(csv_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([filename, leaf_number, leaf_area, pred_sum_leaf, marker_area, pred_sum_marker])

# --- Estatísticas ---
RER_leaf = np.divide(np.absolute(np.subtract(true_leaf, pred_leaf)), true_leaf) * 100
avg_RER_leaf = np.mean(RER_leaf)
std_RER_leaf = np.std(RER_leaf)
print('\n\nAVG RER LEAF: ', avg_RER_leaf, ' STD RER LEAF: ', std_RER_leaf)

RER_marker = np.divide(np.absolute(np.subtract(true_marker, pred_marker)), true_marker) * 100
avg_RER_marker = np.mean(RER_marker)
std_RER_marker = np.std(RER_marker)
print('\nAVG RER MARKER: ', avg_RER_marker, ' STD RER MARKER: ', std_RER_marker)
