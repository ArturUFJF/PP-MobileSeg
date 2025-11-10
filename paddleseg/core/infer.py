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

import collections.abc
from itertools import combinations

import numpy as np
import cv2
import paddle
import paddle.nn.functional as F
from skimage import measure, morphology


def reverse_transform(pred, trans_info, mode='nearest'):
    """recover pred to origin shape"""
    intTypeList = [paddle.int8, paddle.int16, paddle.int32, paddle.int64]
    dtype = pred.dtype
    for item in trans_info[::-1]:
        if isinstance(item[0], list):
            trans_mode = item[0][0]
        else:
            trans_mode = item[0]
        if trans_mode == 'resize':
            h, w = item[1][0], item[1][1]
            if paddle.get_device() == 'cpu' and dtype in intTypeList:
                pred = paddle.cast(pred, 'float32')
                pred = F.interpolate(pred, [h, w], mode=mode)
                pred = paddle.cast(pred, dtype)
            else:
                pred = F.interpolate(pred, [h, w], mode=mode)
        elif trans_mode == 'padding':
            h, w = item[1][0], item[1][1]
            pred = pred[:, :, 0:h, 0:w]
        else:
            raise Exception("Unexpected info '{}' in im_info".format(item[0]))
    return pred


def _mask_area_by_prediction(pred, area_logit, classes=(1, 2)):
    """Apply Hadamard product on area_logit using predicted mask for given classes.

    - pred: paddle.Tensor, either (N,1,H,W) with integer class ids
            or (N,C,H,W) boolean-like (multilabel) mask.
    - area_logit: paddle.Tensor, e.g. (N,1,H,W) or (N,H,W)
    - classes: tuple of class indices to keep (default (1,2)).

    Returns masked area_logit; if shapes are unexpected the original area_logit
    is returned to avoid breaking inference.
    """
    try:
        # Build boolean mask
        if pred.ndim == 4 and pred.shape[1] == 1:
            mask = (pred == classes[0])
            for c in classes[1:]:
                mask = paddle.logical_or(mask, pred == c)
        elif pred.ndim == 4 and pred.shape[1] > 1:
            mask = None
            for c in classes:
                if pred.shape[1] > c:
                    ch = pred[:, c:c+1, :, :]
                    mask = ch if mask is None else paddle.logical_or(mask, ch)
            if mask is None:
                mask = paddle.zeros([pred.shape[0], 1, pred.shape[2], pred.shape[3]], dtype='bool')
        else:
            mask = (pred == classes[0])
            for c in classes[1:]:
                mask = paddle.logical_or(mask, pred == c)

        # Cast mask to area dtype and align shapes
        mask = paddle.cast(mask, area_logit.dtype)

        if mask.shape != area_logit.shape:
            if area_logit.ndim == 3 and mask.ndim == 4 and mask.shape[1] == 1:
                mask = mask.squeeze(1)
            elif area_logit.ndim == 4 and mask.ndim == 3:
                mask = mask.unsqueeze(1)
            elif mask.ndim == 4 and area_logit.ndim == 4 and area_logit.shape[1] == 1 and mask.shape[1] > 1:
                mask = paddle.any(mask, axis=1, keepdim=True).astype(area_logit.dtype)

        return area_logit * mask
    except Exception:
        return area_logit


def flip_combination(flip_horizontal=False, flip_vertical=False):
    """
    Get flip combination.

    Args:
        flip_horizontal (bool): Whether to flip horizontally. Default: False.
        flip_vertical (bool): Whether to flip vertically. Default: False.

    Returns:
        list: List of tuple. The first element of tuple is whether to flip horizontally,
            and the second is whether to flip vertically.
    """

    flip_comb = [(False, False)]
    if flip_horizontal:
        flip_comb.append((True, False))
    if flip_vertical:
        flip_comb.append((False, True))
        if flip_horizontal:
            flip_comb.append((True, True))
    return flip_comb


def stfpm_post_transform(score_map):
    mask = score_map[0][0]
    mask[mask > 0.01] = 1
    mask[mask <= 0.01] = 0
    kernel = morphology.disk(4)
    mask = morphology.opening(mask, kernel)
    mask = paddle.to_tensor(mask[None, None, :, :])
    return paddle.concat([1 - mask, mask], axis=1)


def tensor_flip(x, flip):
    """Flip tensor according directions"""
    if flip[0]:
        x = x[:, :, :, ::-1]
    if flip[1]:
        x = x[:, :, ::-1, :]
    return x


def slide_inference(model, im, crop_size, stride):
    """
    Infer by sliding window.

    Args:
        model (paddle.nn.Layer): model to get logits of image.
        im (Tensor): the input image.
        crop_size (tuple|list). The size of sliding window, (w, h).
        stride (tuple|list). The size of stride, (w, h).

    Return:
        Tensor: The logit of input image.
    """
    h_im, w_im = im.shape[-2:]
    w_crop, h_crop = crop_size
    w_stride, h_stride = stride
    # calculate the crop nums
    rows = int(np.ceil(1.0 * (h_im - h_crop) / h_stride)) + 1
    cols = int(np.ceil(1.0 * (w_im - w_crop) / w_stride)) + 1
    # prevent negative sliding rounds when imgs after scaling << crop_size
    rows = 1 if h_im <= h_crop else rows
    cols = 1 if w_im <= w_crop else cols
    # TODO 'Tensor' object does not support item assignment. If support, use tensor to calculation.
    final_logit = None
    count = None
    num_outputs = None
    for r in range(rows):
        for c in range(cols):
            h1 = r * h_stride
            w1 = c * w_stride
            h2 = min(h1 + h_crop, h_im)
            w2 = min(w1 + w_crop, w_im)
            h1 = max(h2 - h_crop, 0)
            w1 = max(w2 - w_crop, 0)
            im_crop = im[:, :, h1:h2, w1:w2]
            logits = model(im_crop)
            if not isinstance(logits, collections.abc.Sequence):
                raise TypeError(
                    "The type of logits must be one of collections.abc.Sequence, e.g. list, tuple. But received {}"
                    .format(type(logits)))

            # Prepare accumulation containers based on number of outputs
            if num_outputs is None:
                num_outputs = len(logits)
                final_logit = [None] * num_outputs
                count = [np.zeros([1, 1, h_im, w_im]) for _ in range(num_outputs)]

            # Accumulate each output separately
            for j, out in enumerate(logits):
                logit_j = out.numpy()
                if final_logit[j] is None:
                    final_logit[j] = np.zeros([1, logit_j.shape[1], h_im, w_im])
                final_logit[j][:, :, h1:h2, w1:w2] += logit_j[:, :, :h2 - h1, :w2 - w1]
                count[j][:, :, h1:h2, w1:w2] += 1
    if np.sum(count == 0) != 0:
        raise RuntimeError(
            'There are pixel not predicted. It is possible that stride is greater than crop_size'
        )
    # Normalize accumulated logits per output
    if isinstance(final_logit, list):
        for j in range(len(final_logit)):
            final_logit[j] = final_logit[j] / count[j]
        # convert to paddle tensors
        final_logit = [paddle.to_tensor(x) for x in final_logit]
        return final_logit
    else:
        final_logit = final_logit / count
        final_logit = paddle.to_tensor(final_logit)
        return final_logit


def inference(model,
              im,
              trans_info=None,
              is_slide=False,
              stride=None,
              crop_size=None,
              use_multilabel=False):
    """
    Inference for image.

    Args:
        model (paddle.nn.Layer): model to get logits of image.
        im (Tensor): the input image.
        trans_info (list): Image shape informating changed process. Default: None.
        is_slide (bool): Whether to infer by sliding window. Default: False.
        crop_size (tuple|list). The size of sliding window, (w, h). It should be probided if is_slide is True.
        stride (tuple|list). The size of stride, (w, h). It should be probided if is_slide is True.
        use_multilabel (bool, optional): Whether to enable multilabel mode. Default: False.

    Returns:
        Tensor: If ori_shape is not None, a prediction with shape (1, 1, h, w) is returned.
            If ori_shape is None, a logit with shape (1, num_classes, h, w) is returned.
    """
    if hasattr(model, 'data_format') and model.data_format == 'NHWC':
        im = im.transpose((0, 2, 3, 1))
    if not is_slide:
        logits = model(im)
        if not isinstance(logits, collections.abc.Sequence):
            raise TypeError(
                "The type of logits must be one of collections.abc.Sequence, e.g. list, tuple. But received {}"
                .format(type(logits)))

        # If model returns multiple outputs (e.g., [seg_logits, area_logits]), keep
        # them as a sequence so later logic can handle and return the extra output(s).
        if len(logits) == 1:
            logit = logits[0]
        else:
            # keep as list of tensors
            logit = list(logits)
    else:
        logit = slide_inference(model, im, crop_size=crop_size, stride=stride)
    if hasattr(model, 'data_format') and model.data_format == 'NHWC':
        logit = logit.transpose((0, 3, 1, 2))
    if model.__class__.__name__ == 'STFPM':
        logit = stfpm_post_transform(logit)
    if trans_info is not None:
        # If slide_inference returned multiple tensors, logit may be a list
        if isinstance(logit, list):
            # reverse transform first (segmentation) and extra outputs individually
            seg_logit = reverse_transform(logit[0], trans_info, mode='bilinear')
            extra_rets = []
            for extra in logit[1:]:
                extra_rets.append(reverse_transform(extra, trans_info, mode='bilinear'))
            if not use_multilabel:
                pred = paddle.argmax(seg_logit, axis=1, keepdim=True, dtype='int32')
            else:
                pred = (F.sigmoid(seg_logit) > 0.5).astype('int32')
            # Return pred and extra logits (if any). For compatibility return only one extra (first) if exists
            if len(extra_rets) > 0:
                try:
                    masked = _mask_area_by_prediction(pred, extra_rets[0])
                    return pred, masked
                except Exception:
                    return pred, extra_rets[0]
            else:
                return pred, None
        else:
            logit = reverse_transform(logit, trans_info, mode='bilinear')
            if not use_multilabel:
                pred = paddle.argmax(logit, axis=1, keepdim=True, dtype='int32')
            else:
                pred = (F.sigmoid(logit) > 0.5).astype('int32')
            return pred, logit
    else:
        return logit


def aug_inference(model,
                  im,
                  trans_info,
                  scales=1.0,
                  flip_horizontal=False,
                  flip_vertical=False,
                  is_slide=False,
                  stride=None,
                  crop_size=None,
                  use_multilabel=False):
    """
    Infer with augmentation.

    Args:
        model (paddle.nn.Layer): model to get logits of image.
        im (Tensor): the input image.
        trans_info (list): Transforms for image.
        scales (float|tuple|list):  Scales for resize. Default: 1.
        flip_horizontal (bool): Whether to flip horizontally. Default: False.
        flip_vertical (bool): Whether to flip vertically. Default: False.
        is_slide (bool): Whether to infer by sliding wimdow. Default: False.
        crop_size (tuple|list). The size of sliding window, (w, h). It should be probided if is_slide is True.
        stride (tuple|list). The size of stride, (w, h). It should be probided if is_slide is True.
        use_multilabel (bool, optional): Whether to enable multilabel mode. Default: False.

    Returns:
        Tensor: Prediction of image with shape (1, 1, h, w) is returned.
    """
    if isinstance(scales, float):
        scales = [scales]
    elif not isinstance(scales, (tuple, list)):
        raise TypeError(
            '`scales` expects float/tuple/list type, but received {}'.format(
                type(scales)))
    final_logit = None
    h_input, w_input = im.shape[-2], im.shape[-1]
    flip_comb = flip_combination(flip_horizontal, flip_vertical)
    num_augs = len(scales) * len(flip_comb)
    for scale in scales:
        h = int(h_input * scale + 0.5)
        w = int(w_input * scale + 0.5)
        im_scale = F.interpolate(im, [h, w], mode='bilinear')
        for flip in flip_comb:
            im_flip = tensor_flip(im_scale, flip)
            # inference may return (pred, logit) tuple when trans_info is provided inside
            out = inference(model,
                              im_flip,
                              is_slide=is_slide,
                              crop_size=crop_size,
                              stride=stride)
            # inference returns either a tensor or (pred, logit). We need the logit.
            if isinstance(out, tuple) or isinstance(out, list):
                logit = out[1]
            else:
                logit = out
            # logit may be a list (multi-output) or a tensor
            if isinstance(logit, list):
                # flip and resize each
                logit = [tensor_flip(l, flip) for l in logit]
                logit = [F.interpolate(l, [h_input, w_input], mode='bilinear') for l in logit]
                if final_logit is None:
                    final_logit = [0] * len(logit)
                for j in range(len(logit)):
                    final_logit[j] = final_logit[j] + logit[j]
            else:
                logit = tensor_flip(logit, flip)
                logit = F.interpolate(logit, [h_input, w_input], mode='bilinear')
                if final_logit is None:
                    final_logit = 0
                final_logit = final_logit + logit
    # We average the accumulated logits to make the numeric values of `final_logit`
    # comparable to single-scale logits
    final_logit = [l / num_augs for l in final_logit] if isinstance(final_logit, list) else final_logit / num_augs
    final_logit = reverse_transform(final_logit, trans_info, mode='bilinear')
    if isinstance(final_logit, list):
        seg_logit = final_logit[0]
        extra_rets = final_logit[1:]
        if not use_multilabel:
            pred = paddle.argmax(seg_logit, axis=1, keepdim=True, dtype='int32')
        else:
            pred = (F.sigmoid(seg_logit) > 0.5).astype('int32')
        if len(extra_rets) > 0:
            try:
                masked = _mask_area_by_prediction(pred, extra_rets[0])
                return pred, masked
            except Exception:
                return pred, extra_rets[0]
        else:
            return pred, None
    else:
        if not use_multilabel:
            pred = paddle.argmax(final_logit, axis=1, keepdim=True, dtype='int32')
        else:
            pred = (F.sigmoid(final_logit) > 0.5).astype('int32')
        return pred, final_logit
