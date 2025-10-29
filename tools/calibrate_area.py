import argparse
import os
import csv
import json
import numpy as np
import paddle
from paddleseg.cvlibs import Config, SegBuilder
from paddleseg.transforms import Compose
from paddleseg.core import infer
from paddleseg.core.predict import preprocess

# This script computes a linear calibration (real_area = a * raw_pred + b)
# between the raw predicted area sums and the real areas found in area_dict.py.
# It saves `tools/area_calibration.json` with the coefficients and a CSV of per-image
# predictions.

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='model config file')
parser.add_argument('--model_path', required=True, help='trained model params')
parser.add_argument('--list_file', default='data/lsidbeans/train.txt', help='dataset list file (image label area)')
parser.add_argument('--dataset_root', default='.', help='dataset root to join relative paths')
parser.add_argument('--out_csv', default='tools/area_predictions.csv', help='where to write per-image predictions')
parser.add_argument('--out_calib', default='tools/area_calibration.json', help='where to write calibration JSON')
parser.add_argument('--max_samples', type=int, default=1000, help='limit samples for calibration')
args = parser.parse_args()

# build model
cfg = Config(args.config)
builder = SegBuilder(cfg)
model = builder.model
utils = None
# load weights
from paddleseg.utils import utils as _utils
_utils.load_entire_model(model, args.model_path)
model.eval()

# Try to import area_dict
try:
    from area_dict import area_dict
except Exception:
    area_dict = {}

transforms = Compose(builder.val_transforms)

records = []
count = 0
with open(args.list_file, 'r') as f:
    for line in f:
        if count >= args.max_samples:
            break
        items = line.strip().split()
        if len(items) < 1:
            continue
        img_rel = items[0]
        img_path = os.path.join(args.dataset_root, img_rel)
        if not os.path.exists(img_path):
            continue

        data = {'img': img_path}
        data = transforms(data)
        data['img'] = data['img'][np.newaxis, ...]
        data['img'] = paddle.to_tensor(data['img'])

        # get predicted segmentation using infer.inference (restores original size)
        try:
            pred, _ = infer.inference(model, data['img'], trans_info=data['trans_info'], use_multilabel=False)
            pred = paddle.squeeze(pred).numpy().astype('uint8')
        except Exception as e:
            print('segmentation inference failed for', img_path, e)
            continue

        # get area head raw map via forward and reverse_transform
        try:
            logits_list = model(data['img'])
            if isinstance(logits_list, (list, tuple)) and len(logits_list) > 1:
                area_logit = logits_list[1]
                area_logit = infer.reverse_transform(area_logit, data['trans_info'], mode='bilinear')
                area_np = area_logit.numpy()
                if area_np.ndim == 4:
                    area_np = area_np[0]
                if area_np.ndim == 3 and area_np.shape[0] > 1:
                    area_map = area_np[0]
                elif area_np.ndim == 3:
                    area_map = area_np[0]
                else:
                    area_map = area_np
                area_map = np.squeeze(area_map)
            else:
                print('no area head for', img_path)
                continue
        except Exception as e:
            print('area head forward failed for', img_path, e)
            continue

        # compute sums for leaf and square
        mask_leaf = (pred == 1).astype('float32')
        mask_square = (pred == 2).astype('float32')
        pred_leaf_raw = float((area_map * mask_leaf).sum())
        pred_square_raw = float((area_map * mask_square).sum())

        # try to get sample id (3-digit) for lookup in area_dict
        import re
        m = re.search(r"_(\d{3})_", img_rel)
        sample_id = None
        if m:
            sample_id = m.group(1)
        else:
            m2 = re.search(r"(\d{3})", img_rel)
            if m2:
                sample_id = m2.group(1)

        real_area = None
        if sample_id and area_dict:
            real_area = area_dict.get(sample_id)

        records.append((img_rel, sample_id, pred_leaf_raw, pred_square_raw, real_area))
        count += 1

# prepare arrays for regression: use only samples with real_area
leaf_raw = []
real_list = []
for r in records:
    if r[4] is not None:
        leaf_raw.append(r[2])
        real_list.append(r[4])
leaf_raw = np.array(leaf_raw)
real_list = np.array(real_list)

if len(leaf_raw) < 3:
    print('Not enough samples with real area for calibration. Found', len(leaf_raw))
    # still write CSV with raw preds
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image','sample_id','pred_leaf_raw','pred_square_raw','real_area'])
        for r in records:
            writer.writerow(r)
    print('Wrote raw predictions to', args.out_csv)
    exit(0)

# fit linear regression real = a * raw + b
A = np.vstack([leaf_raw, np.ones_like(leaf_raw)]).T
a, b = np.linalg.lstsq(A, real_list, rcond=None)[0]
print('Calibration leaf: real = {:.6f} * raw + {:.6f}'.format(a, b))

# apply calibration to all records
os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
with open(args.out_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image','sample_id','pred_leaf_raw','pred_leaf_calib','real_area','pred_square_raw'])
    for r in records:
        pred_leaf_raw = r[2]
        pred_leaf_calib = a * pred_leaf_raw + b
        writer.writerow([r[0], r[1], pred_leaf_raw, pred_leaf_calib, r[4], r[3]])

# save calibration JSON
calib = {'leaf': {'a': float(a), 'b': float(b)}, 'samples_used': int(len(leaf_raw))}
with open(args.out_calib, 'w') as jf:
    json.dump(calib, jf, indent=2)

print('Wrote calibration to', args.out_calib)
print('Wrote per-image CSV to', args.out_csv)
