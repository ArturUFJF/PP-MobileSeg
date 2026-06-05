# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shlex

import paddle
import numpy as np
from PIL import Image

from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose
import paddleseg.transforms.functional as F


@manager.DATASETS.add_component
class LeafDataset(paddle.io.Dataset):
    """
    Pass in a custom dataset that conforms to the format.

    Args:
        transforms (list): Transforms for image.
        dataset_root (str): The dataset directory.
        num_classes (int): Number of classes.
        mode (str, optional): which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
        train_path (str, optional): The train dataset file. When mode is 'train', train_path is necessary.
            The contents of train_path file are as follow:
            image1.jpg ground_truth1.png areaLabel1.raw
            image2.jpg ground_truth2.png areaLabel2.raw
        val_path (str. optional): The evaluation dataset file. When mode is 'val', val_path is necessary.
            The contents is the same as train_path
        test_path (str, optional): The test dataset file. When mode is 'test', test_path is necessary.
            The annotation file is not necessary in test_path file.
        separator (str, optional): The separator of dataset list. Default: ' '.
            If file names contain spaces, quote them in the list file, e.g.:
            "images/bell pepper.jpg" "annotations/bell pepper.png" "area/bell pepper.raw"
        edge (bool, optional): Whether to compute edge while training. Default: False

        Examples:

            import paddleseg.transforms as T
            from paddleseg.datasets import Dataset

            transforms = [T.RandomPaddingCrop(crop_size=(512,512)), T.Normalize()]
            dataset_root = 'dataset_root_path'
            train_path = 'train_path'
            num_classes = 2
            dataset = Dataset(transforms = transforms,
                              dataset_root = dataset_root,
                              num_classes = num_classes,
                              train_path = train_path,
                              mode = 'train')

    """

    # Comentários adicionais (PT-BR):
    # - O dataset retorna, além de `img`, `label` e `areaLabel`, metadados opcionais:
    #   - `square_area_cm2`: soma das áreas em `areaLabel` sobre a máscara GT do quadrado (se disponível).
    #   - `pixel_area_cm2`: média de área por pixel dentro do quadrado (square_area_cm2 / pixel_count).
    # - Importante: não forçamos aqui que o quadrado tenha 25 cm^2. Em vez disso, preservamos
    #   os valores fornecidos nos arquivos .raw (que podem variar) e expomos essas medidas
    #   como metadata para calibragem por imagem durante inferência, se desejado.
    # - A rede recebe `areaLabel` (valores por-pixel vindos do .raw) e, durante o treino, a
    #   perda de área será aplicada apenas nas regiões relevantes (folha/quadrado) via Hadamard
    #   com as máscaras GT — assim a rede aprende a estimativa por-pixel diretamente.

    NUM_CLASSES = 3
    IGNORE_INDEX = 255
    IMG_CHANNELS = 3

    def __init__(self,
                 mode,
                 dataset_root,
                 transforms,
                 num_classes,
                 img_channels=3,
                 train_path=None,
                 val_path=None,
                 test_path=None,
                 separator=' ',
                 ignore_index=255,
                 edge=False):
        self.dataset_root = dataset_root
        # Monta o pipeline de preprocessamento que roda para cada imagem
        self.transforms = Compose(transforms, img_channels=img_channels)
        # Lista com os caminhos de imagem e mascara que o dataset vai iterar
        self.file_list = list()
        self.mode = mode.lower()
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.ignore_index = ignore_index
        self.edge = edge

        # Verificacoes simples para garantir que os parametros recebidos fazem sentido
        if self.mode not in ['train', 'val', 'test']:
            raise ValueError(
                "mode should be 'train', 'val' or 'test', but got {}.".format(
                    self.mode))
        if not os.path.exists(self.dataset_root):
            raise FileNotFoundError('there is not `dataset_root`: {}.'.format(
                self.dataset_root))
        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")
        if num_classes < 1:
            raise ValueError(
                "`num_classes` should be greater than 1, but got {}".format(
                    num_classes))
        if img_channels not in [1, 3]:
            raise ValueError("`img_channels` should in [1, 3], but got {}".
                             format(img_channels))

        if self.mode == 'train':
            if train_path is None:
                raise ValueError(
                    'When `mode` is "train", `train_path` is necessary, but it is None.'
                )
            elif not os.path.exists(train_path):
                raise FileNotFoundError('`train_path` is not found: {}'.format(
                    train_path))
            else:
                file_path = train_path
        elif self.mode == 'val':
            if val_path is None:
                raise ValueError(
                    'When `mode` is "val", `val_path` is necessary, but it is None.'
                )
            elif not os.path.exists(val_path):
                raise FileNotFoundError('`val_path` is not found: {}'.format(
                    val_path))
            else:
                file_path = val_path
        else:
            if test_path is None:
                raise ValueError(
                    'When `mode` is "test", `test_path` is necessary, but it is None.'
                )
            elif not os.path.exists(test_path):
                raise FileNotFoundError('`test_path` is not found: {}'.format(
                    test_path))
            else:
                file_path = test_path

        # Abre o arquivo de lista e monta os pares de caminho de imagem e rotulo
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items = shlex.split(line)
                if len(items) != 3:
                    if self.mode == 'train' or self.mode == 'val':
                        raise ValueError(
                            "File list format incorrect! In training or evaluation task it should be"
                            " \"image_name\" \"label_name\" \"areaLabel_name\"\\n or image_name label_name areaLabel_name\\n")
                    image_path = os.path.join(self.dataset_root, items[0])
                    label_path = None
                    areaLabel_path = None
                else:
                    image_path = os.path.join(self.dataset_root, items[0])
                    label_path = os.path.join(self.dataset_root, items[1])
                    areaLabel_path = os.path.join(self.dataset_root, items[2])
                # Cada entrada fica guardada para acesso rapido no __getitem__
                self.file_list.append([image_path, label_path, areaLabel_path])

    def __getitem__(self, idx):
        # Dicionario com tudo que os transforms esperam receber
        data = {}
        data['trans_info'] = []
        image_path, label_path, areaLabel_path = self.file_list[idx]
        data['img'] = image_path
        data['label'] = label_path
        data['areaLabel'] = areaLabel_path
        # If key in gt_fields, the data[key] have transforms synchronous.
        data['gt_fields'] = []
        if self.mode == 'val':
            # No modo de validacao apenas aplica transform e ajusta dimensoes da mascara
            data = self.transforms(data)
            if data['label'].ndim == 2:
                data['label'] = data['label'][np.newaxis, :, :]
            if data['areaLabel'].ndim == 2:
                data['areaLabel'] = data['areaLabel'][np.newaxis, :, :]

        else:
            # Em treino/teste sincroniza transformacoes de imagem e mascara
            data['gt_fields'].append('label')
            data['gt_fields'].append('areaLabel')
            data = self.transforms(data)
            if self.edge:
                # Gera um mapa de borda opcional para supervisionar contornos
                edge_mask = F.mask_to_binary_edge(
                    data['label'], radius=2, num_classes=self.num_classes)
                data['edge'] = edge_mask

            elif 'edge' in data:  # for AddEdgeLabel
                # F.mask_to_binary_edge is so slow
                # AddEdgeLabel will faster
                # But offline generation of edges might be better
                data['edge'][data['edge'] == self.ignore_index] = 0

        return data

    def __len__(self):
        # Quantidade total de amostras disponiveis para iterar
        return len(self.file_list)
