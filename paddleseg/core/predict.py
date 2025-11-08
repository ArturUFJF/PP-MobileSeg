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
    area_saved_dir = os.path.join(save_dir, 'area_color_map')

    logger.info("Start to predict...")
    progbar_pred = progbar.Progbar(target=len(img_lists[0]), verbose=1)
    color_map = visualize.get_color_map_list(256, custom_color=custom_color)
    with paddle.no_grad():
        for i, im_path in enumerate(img_lists[local_rank]):
            data = preprocess(im_path, transforms)

            if aug_pred:
                # infer.aug_inference may return (seg_pred, area_pred) or similar tuple.
                pred, area_map = infer.aug_inference(
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
                pred, area_map = infer.inference(
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

            # prepare added image (we save it after we possibly overlay area text)
            added_image = utils.visualize.visualize(
                im_path, pred, color_map, weight=0.6, use_multilabel=use_multilabel)
            added_image_path = os.path.join(added_saved_dir, im_file)
            mkdir(added_image_path)

            # save pseudo color prediction
            pred_mask = utils.visualize.get_pseudo_color_map(
                pred, color_map, use_multilabel=use_multilabel)
            pred_saved_path = os.path.join(
                pred_saved_dir, os.path.splitext(im_file)[0] + ".png")
            mkdir(pred_saved_path)
            pred_mask.save(pred_saved_path)

            # If area_map is available, produce a colorized visualization of the
            # per-pixel area predictions. Map pixel values linearly to [0,255]
            # per-image and apply a perceptual colormap (JET) so tone corresponds
            # to numeric area value.
            try:
                if 'area_map' in locals() and area_map is not None:
                    # area_map may be a paddle Tensor or numpy array. Convert to numpy.
                    if isinstance(area_map, paddle.Tensor):
                        amap = area_map.numpy()
                    else:
                        amap = np.array(area_map)

                    # Squeeze possible channel dims -> (H, W)
                    if amap.ndim == 3 and amap.shape[0] == 1:
                        amap = np.squeeze(amap, axis=0)
                    elif amap.ndim == 4 and amap.shape[0] == 1 and amap.shape[1] == 1:
                        amap = np.squeeze(amap, axis=(0, 1))

                    # Ensure float32
                    amap = amap.astype('float32')

                    # Log diagnostic info to help debugging
                    try:
                        vmin = float(np.nanmin(amap))
                        vmax = float(np.nanmax(amap))
                        logger.info("area_map found for %s: shape=%s dtype=%s min=%f max=%f",
                                    im_file, amap.shape, amap.dtype, vmin, vmax)
                    except Exception:
                        logger.info("area_map found for %s: shape=%s dtype=%s (min/max unavailable)",
                                    im_file, amap.shape, amap.dtype)

                    eps = 1e-6
                    try:
                        denom = (vmax - vmin) if (vmax - vmin) > eps else eps
                        amap_norm = (amap - vmin) / denom
                    except Exception:
                        amap_norm = np.zeros_like(amap)

                    amap_8 = (np.clip(amap_norm, 0.0, 1.0) * 255.0).astype('uint8')
                    # applyColorMap expects BGR output; use JET (good for scalar fields)
                    cmap_bgr = cv2.applyColorMap(amap_8, cv2.COLORMAP_JET)

                    # Compute numeric areas for leaf (class=1) and square (class=2)
                    try:
                        # pred is uint8 numpy array with shape (H, W)
                        leaf_mask = (pred == 1)
                        square_mask = (pred == 2)
                        # Ensure amap shape matches pred
                        if amap.shape != pred.shape:
                            # try transpose or squeeze if necessary
                            amap_proc = np.squeeze(amap)
                        else:
                            amap_proc = amap

                        leaf_area_val = float(np.sum(amap_proc[leaf_mask])) if np.any(leaf_mask) else 0.0
                        square_area_val = float(np.sum(amap_proc[square_mask])) if np.any(square_mask) else 0.0

                        logger.info("Predicted areas for %s: leaf=%.4f  square=%.4f",
                                    im_file, leaf_area_val, square_area_val)

                        # Save numeric results next to area map
                        area_txt_path = os.path.join(
                            area_saved_dir, os.path.splitext(im_file)[0] + "_area.txt")
                        mkdir(area_txt_path)
                        with open(area_txt_path, 'w', encoding='utf-8') as fh:
                            fh.write(f"leaf_area:{leaf_area_val:.6f}\n")
                            fh.write(f"square_area:{square_area_val:.6f}\n")

                        # Overlay text on the added_image for visualization (top-left)
                        try:
                            # added_image is a numpy array (H,W,3) in BGR; draw text top-left
                            text1 = f"Leaf: {leaf_area_val:.2f}"
                            text2 = f"Square: {square_area_val:.2f}"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            scale = 0.8
                            thickness = 2
                            # Put two lines with small margin
                            org1 = (10, 30)
                            org2 = (10, 60)
                            # draw outline for readability
                            cv2.putText(added_image, text1, org1, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                            cv2.putText(added_image, text1, org1, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
                            cv2.putText(added_image, text2, org2, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                            cv2.putText(added_image, text2, org2, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
                        except Exception:
                            logger.exception("Failed overlaying area text on image %s", im_file)
                    except Exception:
                        logger.exception("Failed computing numeric areas for %s", im_file)

                    # Overlay numeric area values onto the color map (bottom-right)
                    try:
                        # Compute text strings
                        text1 = f"Leaf: {leaf_area_val:.2f}"
                        text2 = f"Square: {square_area_val:.2f}"
                        h_c, w_c = cmap_bgr.shape[:2]
                        margin = 10
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        scale = max(0.5, min(w_c, h_c) / 800.0)  # adaptive scale
                        thickness = max(1, int(round(scale * 2)))
                        line_spacing = int(round(6 * scale))

                        lines = [text1, text2]
                        # compute widths/heights
                        sizes = [cv2.getTextSize(l, font, scale, thickness)[0] for l in lines]
                        max_w = max(s[0] for s in sizes)
                        total_h = sum(s[1] for s in sizes) + (len(sizes) - 1) * line_spacing

                        # determine padding and inset so the box doesn't touch edges
                        padding = max(8, int(round(min(w_c, h_c) * 0.01)))
                        inset = max(10, int(round(min(w_c, h_c) * 0.03)))

                        # compute rectangle coordinates: inset from right/bottom
                        rect_right = w_c - margin - inset
                        rect_bottom = h_c - margin - inset
                        rect_left = rect_right - (max_w + 2 * padding)
                        rect_top = rect_bottom - (total_h + 2 * padding)

                        # clamp to image bounds
                        rect_left = max(0, int(rect_left))
                        rect_top = max(0, int(rect_top))
                        rect_right = min(w_c, int(rect_right))
                        rect_bottom = min(h_c, int(rect_bottom))

                        overlay = cmap_bgr.copy()
                        cv2.rectangle(overlay, (rect_left, rect_top), (rect_right, rect_bottom), (0, 0, 0), -1)
                        alpha = 0.60
                        cv2.addWeighted(overlay, alpha, cmap_bgr, 1 - alpha, 0, cmap_bgr)

                        # put each line starting at rect_left + padding, rect_top + padding + first line height
                        x_text = rect_left + padding
                        cur_y = rect_top + padding
                        for idx, l in enumerate(lines):
                            text_h = sizes[idx][1]
                            baseline_y = cur_y + text_h
                            cv2.putText(cmap_bgr, l, (x_text, baseline_y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                            cv2.putText(cmap_bgr, l, (x_text, baseline_y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
                            cur_y += text_h + line_spacing
                    except Exception:
                        logger.exception("Failed overlaying area text on color map for %s", im_file)

                    # Save color map as PNG
                    area_out_path = os.path.join(
                        area_saved_dir, os.path.splitext(im_file)[0] + "_area.png")
                    mkdir(area_out_path)
                    # cv2.imwrite writes BGR correctly
                    cv2.imwrite(area_out_path, cmap_bgr)

                    # Compute numeric areas for leaf (class=1) and square (class=2)
                    try:
                        # pred is uint8 numpy array with shape (H, W)
                        leaf_mask = (pred == 1)
                        square_mask = (pred == 2)
                        # Ensure amap shape matches pred
                        if amap.shape != pred.shape:
                            # try transpose or squeeze if necessary
                            amap_proc = np.squeeze(amap)
                        else:
                            amap_proc = amap

                        leaf_area_val = float(np.sum(amap_proc[leaf_mask])) if np.any(leaf_mask) else 0.0
                        square_area_val = float(np.sum(amap_proc[square_mask])) if np.any(square_mask) else 0.0

                        logger.info("Predicted areas for %s: leaf=%.4f  square=%.4f",
                                    im_file, leaf_area_val, square_area_val)

                        # Save numeric results next to area map
                        area_txt_path = os.path.join(
                            area_saved_dir, os.path.splitext(im_file)[0] + "_area.txt")
                        mkdir(area_txt_path)
                        with open(area_txt_path, 'w', encoding='utf-8') as fh:
                            fh.write(f"leaf_area:{leaf_area_val:.6f}\n")
                            fh.write(f"square_area:{square_area_val:.6f}\n")

                        # Overlay text on the added_image for visualization
                        try:
                            # added_image is a numpy array (H,W,3) in BGR; draw text top-left
                            text1 = f"Leaf: {leaf_area_val:.2f}"
                            text2 = f"Square: {square_area_val:.2f}"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            scale = 0.8
                            thickness = 2
                            # position relative to image size
                            h_img = added_image.shape[0]
                            # Put two lines with small margin
                            org1 = (10, 30)
                            org2 = (10, 60)
                            # draw outline for readability
                            cv2.putText(added_image, text1, org1, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                            cv2.putText(added_image, text1, org1, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
                            cv2.putText(added_image, text2, org2, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                            cv2.putText(added_image, text2, org2, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
                        except Exception:
                            logger.exception("Failed overlaying area text on image %s", im_file)
                    except Exception:
                        logger.exception("Failed computing numeric areas for %s", im_file)
                else:
                    logger.info("No area_map for %s (skipping area visualization)", im_file)
            except Exception as e:
                # Non-fatal: area visualization is optional, but log the exception for debugging
                logger.exception("Failed to generate/save area color map for %s: %s", im_file, str(e))

            # Save the added_image (with optional overlayed area text)
            try:
                cv2.imwrite(added_image_path, added_image)
            except Exception:
                logger.exception("Failed to save added image for %s", im_file)

            progbar_pred.update(i + 1)

    logger.info("Predicted images are saved in {} and {} .".format(
        added_saved_dir, pred_saved_dir))