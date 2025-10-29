# Comentários em PT-BR adicionados para explicar cada parte do arquivo.
# Este arquivo define o modelo PPMobileSeg e sua cabeça de decodificação (PPMobileSegHead)
# para segmentação semântica no PaddlePaddle, seguindo o paper PP-MobileSeg.
# A arquitetura consiste em:
# - Um backbone (extração de características)
# - Uma cabeça de segmentação simples (convolução 1x1 com opção de "depthwise")
# - Um passo de upsample com dois modos: 'intepolate' (bilinear padrão) e 'vim' (otimização para inferência)
# O modelo é registrado no gerenciador de modelos do PaddleSeg para ser construído via config.

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# Você não pode usar este arquivo exceto em conformidade com a Licença.
# Você pode obter uma cópia da Licença em http://www.apache.org/licenses/LICENSE-2.0
# Software fornecido "como está", sem garantias.

import warnings  # Import para avisos; não é utilizado explicitamente abaixo, mas pode ser mantido para extensões

import paddle  # Framework principal (tensores, autograd, etc.)
import paddle.nn as nn  # Módulos de rede neural
import paddle.nn.functional as F  # Funções funcionais (ex.: interpolate)

from paddleseg.cvlibs import manager  # Registro/gerenciamento de componentes (MODELS)
from paddleseg.models import layers  # Camadas utilitárias (não usadas diretamente neste arquivo)
from paddleseg.utils import utils  # Utilidades (ex.: carregamento de pesos)
from paddleseg.models.backbones.strideformer import ConvBNAct  # Bloco conv->BN->Ativação pronto

# O decorador abaixo registra a classe no registry de modelos do PaddleSeg sob o nome da classe.
@manager.MODELS.add_component
class PPMobileSeg(nn.Layer):
    """
    Implementação do PP_MobileSeg baseada em PaddlePaddle.
    Referência: https://arxiv.org/abs/2304.05152

    Args:
        num_classes (int): Número de classes alvo.
        backbone (nn.Layer): Backbone de extração de features (deve definir feat_channels).
        head_use_dw (bool, opcional): Se a cabeça usa convoluções "depthwise" (via groups).
        align_corners (bool, opcional): Parâmetro do resize bilinear para alinhamento de cantos.
        pretrained (str, opcional): Caminho/URL de pesos pré-treinados para carregar no modelo.
        upsample (str, opcional): Tipo de upsample. 'intepolate' (padrão) ou 'vim' para otimização.
                                 Obs.: 'intepolate' aqui é uma grafia mantida pelo código de origem.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 head_use_dw=True,
                 align_corners=False,
                 pretrained=None,
                 upsample='intepolate'):
        super().__init__()
        # Guarda referências e hiperparâmetros
        self.backbone = backbone  # Backbone deve retornar um mapa de features compatível com a cabeça
        self.upsample = upsample  # Modo de upsample ('intepolate' padrão ou 'vim' otimizado)
        self.num_classes = num_classes

        # Cria a cabeça de decodificação (converte features do backbone em logits por classe)
        # in_channels vem do backbone.feat_channels[0] (convenciona-se que seja o mapa principal)
        self.decode_head = PPMobileSegHead(
            num_classes=num_classes,
            in_channels=backbone.feat_channels[0],
            use_dw=head_use_dw,
            align_corners=align_corners)
        
        self.area_head = AreaSegHead(
            num_classes=num_classes,  
            in_channels=backbone.feat_channels[0],
            use_dw=head_use_dw,
            align_corners=align_corners)

        self.align_corners = align_corners  # Propagado para F.interpolate
        self.pretrained = pretrained  # Caminho/URL de pesos pré-treinados
        self.init_weight()  # Carrega pesos, se informados

    def init_weight(self):
        # Carrega todo o estado do modelo se self.pretrained for fornecido
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        # x: tensor de entrada (B, C, H, W)
        x_hw = x.shape[2:]  # Guarda a resolução original para upsample posterior
        x = self.backbone(x)  # Extrai features com o backbone
        seg_logits = self.decode_head(x)  # Converte features em logits por classe (B, num_classes, h, w)
        area_logits = self.area_head(x)  # Converte features em logits por classe (B, num_classes, h, w)

        # Estratégia de upsample:
        # - Durante treino (self.training == True), sempre usa interpolate bilinear.
        # - Também usa interpolate se upsample == 'intepolate' (padrão) ou num_classes < 30 (heurística).
        if self.upsample == 'intepolate' or self.training or self.num_classes < 30:
            seg_logits = F.interpolate(
                seg_logits, x_hw, mode='bilinear', align_corners=self.align_corners)
            area_logits = F.interpolate(
                area_logits, x_hw, mode='bilinear', align_corners=self.align_corners)
        else:
            # Caso seja passado um modo de upsample não implementado
            raise NotImplementedError(self.upsample, " is not implemented")

        # Retorno como lista, seguindo a convenção do PaddleSeg (permite múltiplas saídas)
        return [seg_logits, area_logits]

    def loss_computation(self, logits_list, losses, data):
        """
        Custom loss computation for PPMobileSeg.

        This method expects `logits_list` to be [seg_logits, area_logits].
        It computes:
          - segmentation loss using data['label'] and losses['types'][0]
          - two area losses (leaf and square) by masking the area labels using
            the predicted segmentation map and then computes the area loss
            (using losses['types'][1]) inside each masked region. The two
            area losses are summed and multiplied by losses['coef'][1].

        Returns a list [seg_loss, area_loss_total] so it matches the current
        YAML config which defines two loss types/coefs.
        """
        if not isinstance(logits_list, (list, tuple)) or len(logits_list) < 2:
            raise RuntimeError(
                "logits_list must be a list or tuple with at least two elements: [seg_logits, area_logits].")

        seg_logits = logits_list[0]
        area_logits = logits_list[1]

        # Extract labels from data dict
        labels = data.get('label', None)
        area_labels = data.get('areaLabel', None)
        if labels is None or area_labels is None:
            raise RuntimeError("data must contain 'label' and 'areaLabel' for loss computation")

        # Remove channel dim if present: (N,1,H,W) -> (N,H,W)
        if labels.ndim == 4 and labels.shape[1] == 1:
            labels = paddle.squeeze(labels, axis=1)
        if area_labels.ndim == 4 and area_labels.shape[1] == 1:
            area_labels = paddle.squeeze(area_labels, axis=1)

        labels = labels.astype('int64')
        area_labels = area_labels.astype('int64')

        # Default coefficients
        coef_seg = losses['coef'][0] if 'coef' in losses and len(losses['coef']) > 0 else 1.0
        coef_area = losses['coef'][1] if 'coef' in losses and len(losses['coef']) > 1 else 1.0

        # Segmentation loss (apply first loss function to seg_logits)
        seg_loss_fn = losses['types'][0]
        seg_loss = seg_loss_fn(seg_logits, labels) * coef_seg

        # Build predicted segmentation map to create masks (use argmax)
        seg_pred = paddle.argmax(seg_logits, axis=1)  # (N, H, W)

        # Define class ids for leaf and square (adjust if your labels differ)
        leaf_class = 1
        square_class = 2

        ignore_index = 255

        # Prepare masked area labels for leaf
        leaf_mask = (seg_pred == leaf_class)  # boolean mask
        masked_leaf_labels = area_labels.clone()
        # set outside-leaf pixels to ignore_index
        masked_leaf_labels[~leaf_mask] = ignore_index

        # Prepare masked area labels for square
        square_mask = (seg_pred == square_class)
        masked_square_labels = area_labels.clone()
        masked_square_labels[~square_mask] = ignore_index

        # Area loss: compute masked MSE manually (ignore pixels == ignore_index)
        # area_logits may have multiple channels; use the first channel as scalar prediction
        # Ensure tensors are float32 for subtraction
        area_pred = area_logits
        # If area_pred has multiple channels, reduce to one channel (mean or first)
        if area_pred.ndim == 4 and area_pred.shape[1] > 1:
            # take first channel as the scalar area prediction
            area_pred = area_pred[:, :1, :, :]

        # Prepare masked labels as float32 and add channel dim to match area_pred
        masked_leaf = paddle.cast(masked_leaf_labels, 'float32')
        masked_square = paddle.cast(masked_square_labels, 'float32')
        if masked_leaf.ndim == 3:
            masked_leaf = paddle.unsqueeze(masked_leaf, axis=1)
        if masked_square.ndim == 3:
            masked_square = paddle.unsqueeze(masked_square, axis=1)

        # Create valid pixel masks (1.0 for valid, 0.0 for ignore_index)
        valid_leaf = paddle.cast(masked_leaf != ignore_index, 'float32')
        valid_square = paddle.cast(masked_square != ignore_index, 'float32')

        # Compute masked MSE: sum((pred - target)^2 * valid) / (sum(valid) + eps)
        eps = 1e-6
        # leaf
        diff_leaf = area_pred - masked_leaf
        sq_leaf = paddle.square(diff_leaf) * valid_leaf
        denom_leaf = paddle.sum(valid_leaf)
        leaf_loss = paddle.sum(sq_leaf) / (denom_leaf + eps)
        # square
        diff_square = area_pred - masked_square
        sq_square = paddle.square(diff_square) * valid_square
        denom_square = paddle.sum(valid_square)
        square_loss = paddle.sum(sq_square) / (denom_square + eps)

        # If there are no valid pixels for a mask, its loss will be near 0 due to eps.
        area_loss = (leaf_loss + square_loss) * coef_area

        return [seg_loss, area_loss]


class PPMobileSegHead(nn.Layer): #decoder aqui
    # Cabeça simples de segmentação:
    # - Um bloco Conv+BN+ReLU (linear_fuse) com kernel 1x1
    #   Opcionalmente "depthwise" via groups=in_channels (opera canal a canal)
    # - Dropout 2D
    # - Convolução 1x1 final para produzir logits de num_classes
    def __init__(self,
                 num_classes,
                 in_channels,
                 use_dw=False,
                 dropout_ratio=0.1,
                 align_corners=False):
        super().__init__()
        self.align_corners = align_corners  # Não é usado diretamente aqui, mas mantido por consistência
        self.last_channels = in_channels  # Número de canais das features do backbone

        # Bloco de fusão linear:
        # ConvBNAct com kernel 1x1. Se use_dw=True, usa groups=in_channels (convolução por canal),
        # que é uma operação leve (sem mistura entre canais). Caso contrário, é conv 1x1 padrão.
        self.linear_fuse = ConvBNAct(
            in_channels=self.last_channels,
            out_channels=self.last_channels,
            kernel_size=1,
            stride=1,
            groups=self.last_channels if use_dw else 1,
            act=nn.ReLU)
        # Regularização para reduzir overfitting
        self.dropout = nn.Dropout2D(dropout_ratio)
        # Projeção final para o espaço de classes (logits por classe)
        self.conv_seg = nn.Conv2D(
            self.last_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # x aqui é o mapa de features do backbone (espera-se tensor 4D)
        x = self.linear_fuse(x)  # Ajuste/normalização das features com activação ReLU
        x = self.dropout(x)  # Dropout espacial
        x = self.conv_seg(x)  # Logits por classe (B, num_classes, h, w)
        return x
    
    
class AreaSegHead(nn.Layer): #decoder aqui
    # Cabeça simples de segmentação:
    # - Um bloco Conv+BN+ReLU (linear_fuse) com kernel 1x1
    #   Opcionalmente "depthwise" via groups=in_channels (opera canal a canal)
    # - Dropout 2D
    # - Convolução 1x1 final para produzir logits de num_classes
    def __init__(self,
                 num_classes,
                 in_channels,
                 use_dw=False,
                 dropout_ratio=0.1,
                 align_corners=False):
        super().__init__()
        self.align_corners = align_corners  # Não é usado diretamente aqui, mas mantido por consistência
        self.last_channels = in_channels  # Número de canais das features do backbone

        # Bloco de fusão linear:
        # ConvBNAct com kernel 1x1. Se use_dw=True, usa groups=in_channels (convolução por canal),
        # que é uma operação leve (sem mistura entre canais). Caso contrário, é conv 1x1 padrão.
        self.linear_fuse = ConvBNAct(
            in_channels=self.last_channels,
            out_channels=self.last_channels,
            kernel_size=1,
            stride=1,
            groups=self.last_channels if use_dw else 1,
            act=nn.ReLU)
        # Regularização para reduzir overfitting
        self.dropout = nn.Dropout2D(dropout_ratio)
        # Projeção final para o espaço de classes (logits por classe)
        self.conv_seg = nn.Conv2D(
            self.last_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # x aqui é o mapa de features do backbone (espera-se tensor 4D)
        x = self.linear_fuse(x)  # Ajuste/normalização das features com activação ReLU
        x = self.dropout(x)  # Dropout espacial
        x = self.conv_seg(x)  # Logits por classe (B, num_classes, h, w)
        return x
    
    #Lembrar do produto de Hadamard


def loss_computation(self, logits_list, losses, data):
    # logits_list[0] -> seg_logits, logits_list[1] -> area_logits
    seg_logits = logits_list[0]
    area_logits = logits_list[1]

    # extrair labels do data
    labels = data['label']
    area_labels = data['areaLabel']

    # squeeze se necessário
    if labels.ndim == 4 and labels.shape[1] == 1:
        labels = paddle.squeeze(labels, axis=1)
    if area_labels.ndim == 4 and area_labels.shape[1] == 1:
        area_labels = paddle.squeeze(area_labels, axis=1)

    labels = labels.astype('int64')
    area_labels = area_labels.astype('int64')

    # 1) gerar máscara de interesse a partir de seg_logits (predição)
    seg_pred = paddle.argmax(seg_logits, axis=1)  # (N, H, W), ints
    # Caso queira usar GT: seg_pred = labels

    # 2) criar máscaras booleanas
    leaf_class = 1  # ajuste conforme sua label (leaf)
    square_class = 2  # ajuste conforme sua label (square)
    leaf_mask = (seg_pred == leaf_class)        # bool tensor
    square_mask = (seg_pred == square_class)    # bool tensor

    # 3) transformar area_labels em ignore_index fora da máscara
    ignore_index = 255
    masked_area_labels = area_labels.clone()
    masked_area_labels[~leaf_mask] = ignore_index
    masked_area_labels[~square_mask] = ignore_index

    # 4) calcular losses — usar losses['types'] na ordem
    loss_list = []
    for i, loss_fn in enumerate(losses['types']):
        coef = losses['coef'][i] if 'coef' in losses and len(losses['coef']) > i else 1.0
        if i == 0:
            # seg loss com labels completos
            loss_list.append(coef * loss_fn(seg_logits, labels))
        elif i == 1:
            # area loss apenas onde mask == True (labels fora são ignore_index)
            loss_list.append(coef * loss_fn(area_logits, masked_area_labels))
        else:
            loss_list.append(coef * loss_fn(logits_list[i], labels))
    return loss_list