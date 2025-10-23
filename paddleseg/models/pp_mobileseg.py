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
    area_num_classes (int, opcional): Número de canais previstos pela cabeça de área supervisionada.
    area_head_use_dw (bool, opcional): Permite configurar depthwise apenas na cabeça de área.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 head_use_dw=True,
                 align_corners=False,
                 pretrained=None,
                 upsample='intepolate',
                 area_num_classes=None,
                 area_head_use_dw=None):
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

        # Cabeça opcional para a máscara de área supervisionada.
        self.area_head = None
        if area_num_classes is not None:
            use_dw_area = head_use_dw if area_head_use_dw is None else area_head_use_dw
            self.area_head = AreaSegHead(
                num_classes=area_num_classes,
                in_channels=backbone.feat_channels[0],
                use_dw=use_dw_area,
                align_corners=align_corners)

        self.align_corners = align_corners  # Propagado para F.interpolate
        self.pretrained = pretrained  # Caminho/URL de pesos pré-treinados
        self.init_weight()  # Carrega pesos, se informados

    def init_weight(self):
        # Carrega todo o estado do modelo se self.pretrained for fornecido
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def _upsample_segmentation(self, logits, target_hw):
        """Aplica a estratégia de upsample original para a cabeça principal."""
        if logits is None:
            return None

        if self.upsample == 'intepolate' or self.training or self.num_classes < 30:
            return F.interpolate(
                logits, target_hw, mode='bilinear', align_corners=self.align_corners)

        if self.upsample == 'vim':
            labelset = paddle.unique(paddle.argmax(logits, 1))
            reduced = paddle.gather(logits, labelset, axis=1)
            reduced = F.interpolate(
                reduced, target_hw, mode='bilinear', align_corners=self.align_corners)

            pred = paddle.argmax(reduced, 1)
            pred_retrieve = paddle.zeros(pred.shape, dtype='int32')
            for i, val in enumerate(labelset):
                pred_retrieve[pred == i] = labelset[i].cast('int32')
            return pred_retrieve

        raise NotImplementedError(self.upsample, " is not implemented")

    def forward(self, x):
        # x: tensor de entrada (B, C, H, W)
        x_hw = x.shape[2:]  # Guarda a resolução original para upsample posterior
        feats = self.backbone(x)  # Extrai features com o backbone
        seg_logits = self.decode_head(feats)  # Logits por classe (B, num_classes, h, w)

        area_logits = None
        hadamard_logits = None
        if self.area_head is not None:
            area_logits = self.area_head(feats)
            if area_logits.shape[1] not in (1, seg_logits.shape[1]):
                raise ValueError(
                    "`area_head` deve gerar 1 canal ou o mesmo número de canais da cabeça principal.")
            # Produto de Hadamard (element-wise) entre as duas previsões.
            hadamard_logits = seg_logits * area_logits

        seg_out = self._upsample_segmentation(seg_logits, x_hw)

        outputs = [seg_out]
        if area_logits is not None:
            area_out = F.interpolate(
                area_logits,
                x_hw,
                mode='bilinear',
                align_corners=self.align_corners)
            hadamard_out = F.interpolate(
                hadamard_logits,
                x_hw,
                mode='bilinear',
                align_corners=self.align_corners)
            outputs.extend([area_out, hadamard_out])

        return outputs


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
    
    #realizar o Produto de Hadamard entre o mapa de predição e o mapa de área supervisionada