# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math

import cv2
import numpy as np
import paddle

from paddleseg import utils
from paddleseg.core import infer
from paddleseg.utils import logger, progbar, visualize

# optional area dictionary (user-provided) to compare real areas
try:
    from area_dict import area_dict
except Exception:
    area_dict = {}
import json
import os

# load calibration if available (tools/area_calibration.json)
calib_path = os.path.join(os.getcwd(), 'tools', 'area_calibration.json')
calib_params = None
if os.path.exists(calib_path):
    try:
        with open(calib_path, 'r') as f:
            calib_params = json.load(f)
    except Exception:
        calib_params = None


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


def partition_list(arr, m):
    """split the list 'arr' into m pieces"""
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


def preprocess(im_path, transforms):
    data = {}
    data['img'] = im_path
    data = transforms(data)
    data['img'] = data['img'][np.newaxis, ...]
    data['img'] = paddle.to_tensor(data['img'])
    return data


def predict(model,
            model_path,
            transforms,
            image_list,
            image_dir=None,
            save_dir='output',
            aug_pred=False,
            scales=1.0,
            flip_horizontal=True,
            flip_vertical=False,
            is_slide=False,
            stride=None,
            crop_size=None,
            custom_color=None,
            use_multilabel=False):
    """
    predict and visualize the image_list.

    Args:
        model (nn.Layer): Used to predict for input image.
        model_path (str): The path of pretrained model.
        transforms (transform.Compose): Preprocess for input image.
        image_list (list): A list of image path to be predicted.
        image_dir (str, optional): The root directory of the images predicted. Default: None.
        save_dir (str, optional): The directory to save the visualized results. Default: 'output'.
        aug_pred (bool, optional): Whether to use mulit-scales and flip augment for predition. Default: False.
        scales (list|float, optional): Scales for augment. It is valid when `aug_pred` is True. Default: 1.0.
        flip_horizontal (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_pred` is True. Default: True.
        flip_vertical (bool, optional): Whether to use flip vertically augment. It is valid when `aug_pred` is True. Default: False.
        is_slide (bool, optional): Whether to predict by sliding window. Default: False.
        stride (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        crop_size (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        custom_color (list, optional): Save images with a custom color map. Default: None, use paddleseg's default color map.
        use_multilabel (bool, optional): Whether to enable multilabel mode. Default: False.

    """
    utils.utils.load_entire_model(model, model_path)
    model.eval()
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    if nranks > 1:
        img_lists = partition_list(image_list, nranks)
    else:
        img_lists = [image_list]

    added_saved_dir = os.path.join(save_dir, 'added_prediction')
    pred_saved_dir = os.path.join(save_dir, 'pseudo_color_prediction')

    logger.info("Start to predict...")
    progbar_pred = progbar.Progbar(target=len(img_lists[0]), verbose=1)
    color_map = visualize.get_color_map_list(256, custom_color=custom_color)
    with paddle.no_grad():
        for i, im_path in enumerate(img_lists[local_rank]):
            data = preprocess(im_path, transforms)

            if aug_pred:
                pred, _ = infer.aug_inference(
                    model,
                    data['img'],
                    trans_info=data['trans_info'],
                    scales=scales,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size,
                    use_multilabel=use_multilabel)
            else:
                pred, _ = infer.inference(
                    model,
                    data['img'],
                    trans_info=data['trans_info'],
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size,
                    use_multilabel=use_multilabel)
            pred = paddle.squeeze(pred)
            pred = pred.numpy().astype('uint8')

            # get the saved name
            if image_dir is not None:
                im_file = im_path.replace(image_dir, '')
            else:
                im_file = os.path.basename(im_path)
            if im_file[0] == '/' or im_file[0] == '\\':
                im_file = im_file[1:]

            # save added image (with optional area overlay)
            added_image = utils.visualize.visualize(
                im_path, pred, color_map, weight=0.6, use_multilabel=use_multilabel)

            # Try to obtain area prediction from the model's second head (if present).
            # We run a direct forward pass to get all logits, then reverse-transform the
            # area-logits to original image size and compute per-class areas via Hadamard
            # product between area map and binary masks from `pred`.
            area_info = {}
            try:
                logits_list = model(data['img'])
                # Diagnostics: log what the model returned so user can see why area may be missing
                try:
                    logger.debug(f"predict: model returned {type(logits_list)}")
                except Exception:
                    pass

                if not isinstance(logits_list, (list, tuple)):
                    # Model returned only a single head (segmentation). No area head available.
                    logger.info("predict: model returned a single output; no area head detected.")
                elif len(logits_list) <= 1:
                    logger.info("predict: model outputs sequence but has no second head for area.")
                else:
                    area_logit = logits_list[1]
                    # Ensure area_logit is a paddle Tensor (some models might return numpy)
                    if not isinstance(area_logit, paddle.Tensor):
                        try:
                            area_logit = paddle.to_tensor(area_logit)
                        except Exception:
                            logger.warning("predict: area head returned non-tensor and couldn't be converted; skipping area.")
                            raise

                    # Reverse transform to original image size
                    try:
                        area_logit = infer.reverse_transform(area_logit, data['trans_info'], mode='bilinear')
                    except Exception as e:
                        logger.warning(f"predict: reverse_transform on area_logit failed: {e}")
                        raise

                    area_np = area_logit.numpy()
                    # Normalize array dims to (H, W) taking first channel if needed
                    if area_np.ndim == 4:
                        area_np = area_np[0]
                    if area_np.ndim == 3:
                        area_map = area_np[0]
                    else:
                        area_map = area_np

                    leaf_class = 1
                    square_class = 2
                    mask_leaf = (pred == leaf_class).astype('float32')
                    mask_square = (pred == square_class).astype('float32')
                    area_map = np.squeeze(area_map).astype('float32')
                    pred_area_leaf = float((area_map * mask_leaf).sum())
                    pred_area_square = float((area_map * mask_square).sum())
                    # raw sums (model output units)
                    area_info['pred_leaf'] = pred_area_leaf
                    area_info['pred_square'] = pred_area_square

                    # Also compute pixel counts of predicted masks (these are in original image pixels)
                    pixel_count_leaf = float(mask_leaf.sum())
                    pixel_count_square = float(mask_square.sum())
                    area_info['pred_leaf_pixel_count'] = pixel_count_leaf
                    area_info['pred_square_pixel_count'] = pixel_count_square

                    # Per-image calibration using square marker area measured by pixel counts.
                    # The square is 5cm x 5cm = 25 cm^2. Compute pixel_area_cm2 = 25.0 / pixel_count_square
                    # and then estimate leaf area as pixel_count_leaf * pixel_area_cm2.
                    try:
                        eps = 1e-8
                        if pixel_count_square > eps:
                            pixel_area_cm2 = 25.0 / pixel_count_square
                            area_info['pixel_area_cm2'] = pixel_area_cm2
                            area_info['pred_leaf_calib_per_image_pixels'] = pixel_count_leaf * pixel_area_cm2
                            area_info['pred_square_calib_per_image_pixels'] = pixel_count_square * pixel_area_cm2
                        else:
                            area_info['pixel_area_cm2'] = None
                            area_info['pred_leaf_calib_per_image_pixels'] = None
                            area_info['pred_square_calib_per_image_pixels'] = None
                    except Exception:
                        area_info['pixel_area_cm2'] = None
                        area_info['pred_leaf_calib_per_image_pixels'] = None
                        area_info['pred_square_calib_per_image_pixels'] = None

                    # Also keep previous per-image calibration based on area_map sums (if useful)
                    try:
                        if pred_area_square > eps:
                            per_image_scale = 25.0 / pred_area_square
                            area_info['pred_leaf_calib_per_image'] = per_image_scale * pred_area_leaf
                            area_info['pred_square_calib_per_image'] = per_image_scale * pred_area_square
                            area_info['per_image_scale'] = per_image_scale
                        else:
                            area_info['pred_leaf_calib_per_image'] = None
                            area_info['pred_square_calib_per_image'] = None
                            area_info['per_image_scale'] = None
                    except Exception:
                        area_info['pred_leaf_calib_per_image'] = None
                        area_info['pred_square_calib_per_image'] = None
                        area_info['per_image_scale'] = None

                    # apply global calibration if available (leaf/square)
                    if calib_params is not None:
                        try:
                            if 'leaf' in calib_params:
                                a = float(calib_params['leaf'].get('a', 1.0))
                                b = float(calib_params['leaf'].get('b', 0.0))
                                area_info['pred_leaf_calib'] = a * pred_area_leaf + b
                            if 'square' in calib_params:
                                a2 = float(calib_params['square'].get('a', 1.0))
                                b2 = float(calib_params['square'].get('b', 0.0))
                                area_info['pred_square_calib'] = a2 * pred_area_square + b2
                        except Exception:
                            logger.warning("predict: failed to apply calibration parameters")

                    # attempt to extract sample id from filename to lookup real area
                    m = __import__('re').search(r"_(\d{3})_", im_file)
                    sample_id = None
                    if m:
                        sample_id = m.group(1)
                    else:
                        m2 = __import__('re').search(r"(\d{3})", im_file)
                        if m2:
                            sample_id = m2.group(1)

                    if sample_id and area_dict:
                        real_area = area_dict.get(sample_id)
                        if real_area is not None:
                            area_info['real'] = real_area
            except Exception:
                # Keep area_info empty on any error; we've already logged details above.
                area_info = {}

            # overlay area text (if available) on the added_image at top-right
            try:
                img_h, img_w = added_image.shape[:2]
                lines = []
                # prefer per-image pixel-based calibration (derived from square marker) >
                # per-image area_map-based calibration > global calib > raw
                if 'pred_leaf_calib_per_image_pixels' in area_info and area_info.get('pred_leaf_calib_per_image_pixels') is not None:
                    lines.append(f"Área prevista (folha): {area_info['pred_leaf_calib_per_image_pixels']:.3f} cm^2")
                elif 'pred_leaf_calib_per_image' in area_info and area_info.get('pred_leaf_calib_per_image') is not None:
                    lines.append(f"Área prevista (folha): {area_info['pred_leaf_calib_per_image']:.3f} cm^2")
                elif 'pred_leaf_calib' in area_info:
                    lines.append(f"Área prevista (folha): {area_info['pred_leaf_calib']:.3f} cm^2")
                elif 'pred_leaf' in area_info:
                    lines.append(f"Área prevista (folha) [raw]: {area_info['pred_leaf']:.3f}")
                if 'pred_square_calib_per_image_pixels' in area_info and area_info.get('pred_square_calib_per_image_pixels') is not None:
                    lines.append(f"Área prevista (quadrado): {area_info['pred_square_calib_per_image_pixels']:.3f} cm^2")
                elif 'pred_square_calib_per_image' in area_info and area_info.get('pred_square_calib_per_image') is not None:
                    lines.append(f"Área prevista (quadrado): {area_info['pred_square_calib_per_image']:.3f} cm^2")
                elif 'pred_square_calib' in area_info:
                    lines.append(f"Área prevista (quadrado): {area_info['pred_square_calib']:.3f} cm^2")
                elif 'pred_square' in area_info:
                    lines.append(f"Área prevista (quadrado) [raw]: {area_info['pred_square']:.3f}")
                if 'real' in area_info:
                    lines.append(f"Área real: {area_info['real']}")

                if lines:
                    margin = 8
                    pad = 6
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 1
                    text_sizes = [cv2.getTextSize(l, font, font_scale, thickness)[0] for l in lines]
                    block_w = max(w for (w, h) in text_sizes) + pad * 2
                    block_h = sum(h for (w, h) in text_sizes) + pad * 2 + (len(lines) - 1) * 4
                    x1 = img_w - block_w - margin
                    y1 = margin
                    x2 = img_w - margin
                    y2 = margin + block_h
                    overlay = added_image.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
                    alpha = 0.5
                    added_image = cv2.addWeighted(overlay, alpha, added_image, 1 - alpha, 0)
                    y_text = y1 + pad + text_sizes[0][1]
                    for idx, l in enumerate(lines):
                        cv2.putText(added_image, l, (x1 + pad, y_text), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                        if idx + 1 < len(text_sizes):
                            y_text += text_sizes[idx + 1][1] + 4
            except Exception:
                pass

            added_image_path = os.path.join(added_saved_dir, im_file)
            mkdir(added_image_path)
            cv2.imwrite(added_image_path, added_image)

            # save pseudo color prediction
            pred_mask = utils.visualize.get_pseudo_color_map(
                pred, color_map, use_multilabel=use_multilabel)
            pred_saved_path = os.path.join(
                pred_saved_dir, os.path.splitext(im_file)[0] + ".png")
            mkdir(pred_saved_path)
            pred_mask.save(pred_saved_path)

            progbar_pred.update(i + 1)

    logger.info("Predicted images are saved in {} and {} .".format(
        added_saved_dir, pred_saved_dir))