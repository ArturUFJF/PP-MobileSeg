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

"""
Comentários gerais (PT-BR):

Este módulo implementa o modelo PPMobileSeg adaptado para duas saídas:
- `seg_logits`: logits de segmentação (num_classes canais) usados para treinar/avaliar a segmentação.
- `area_logits`: mapa de regressão por-pixel (1 canal) que estima a área associada a cada pixel
    (por exemplo, área estimada em unidades arbitrárias fornecidas pelos arquivos .raw).

Principais decisões implementadas aqui:
- A cabeça de área (`AreaSegHead`) devolve um mapa de 1 canal com valores contínuos.
- A função `loss_computation` calcula duas perdas:
    1) perda de segmentação (CrossEntropy) usando `data['label']`;
    2) perda de área (MSE) calculada apenas dentro das máscaras dos objetos (folha e quadrado)
         usando o produto de Hadamard entre a máscara binária (GT) e os valores de `areaLabel` (.raw).

Isso faz com que a rede aprenda os valores por-pixel da mapagem de área diretamente dos .raw,
sem forçar uma área fixa para o marcador (quadrado). A calibração absoluta (cm^2) pode ser
tratada separadamente no passo de inferência/visualização se necessário.
"""

import warnings  # Import para avisos; não é utilizado explicitamente abaixo, mas pode ser mantido para extensões

import paddle  # Framework principal (tensores, autograd, etc.)
import paddle.nn as nn  # Módulos de rede neural
import paddle.nn.functional as F  # Funções funcionais (ex.: interpolate)

from paddleseg.core.train import check_logits_losses
from paddleseg.cvlibs import manager  # Registro/gerenciamento de componentes (MODELS)
from paddleseg.models import layers, losses  # Camadas utilitárias (não usadas diretamente neste arquivo)
from paddleseg.utils import utils  # Utilidades (ex.: carregamento de pesos)
from paddleseg.models.backbones.strideformer import ConvBNAct  # Bloco conv->BN->Ativação pront

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
            num_classes=1,  
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
        
        #pred = seg_logits.argmax(axis=1, keepdim=True)  # (B, 1, H, W)
        #mask = ((pred == 1) | (pred == 2)).astype('float32')  # Máscara binária para classes de interesse
        #area_logits = area_logits * mask  # Aplica a máscara ao mapa de área, produto de Hadamard.
        #Acima não está correto, o correto é fazer isso apenas na loss_computation

        # Retorno como lista, seguindo a convenção do PaddleSeg (permite múltiplas saídas)
        return [seg_logits, area_logits]

    def loss_computation(self, logits_list, losses, data):
        #Perda da segmentação
        seg_logits, area_logits = logits_list
        seg_labels = data['label'].astype('int64')
        crossEntropy = losses['types'][0]
        coef_ce = losses['coef'][0]
        seg_loss = crossEntropy(seg_logits, seg_labels)

        #Perda da área (MSE com máscara)
        # 1. MÁSCARAS BINÁRIAS DAS PREVISÕES
        leaf_mask = (seg_logits.argmax(axis=1, keepdim=True) == 1).astype('float32')
        square_mask = (seg_logits.argmax(axis=1, keepdim=True) == 2).astype('float32')

        # 2. GABARITO (Ground Truth)
        area_gt = data['areaLabel'].astype('float32')
        if area_gt.ndim == 3:
            area_gt = area_gt.unsqueeze(1)
    
        coef_mse = losses['coef'][1]
    
        # 3. ZERAR ERRO ONDE NÃO HÁ INTERESSE, USANDO AS MÁSCARAS DA PREVISÃO
        # Produto de Hadamard para manter erros apenas nas regiões de interesse
        leaf_error = (area_logits - area_gt) * leaf_mask
        square_error = (area_logits - area_gt) * square_mask
    
        leaf_error_sq = paddle.square(leaf_error)
        square_error_sq = paddle.square(square_error)

        # 4. CALCULAR A MÉDIA CORRETAMENTE (Soma / Contagem)
        epsilon = 1e-6 # Evitar divisão por zero
    
        # Contar quantos pixels foram previstos como folha/quadrado
        leaf_pixel_count = paddle.sum(leaf_mask) + epsilon
        square_pixel_count = paddle.sum(square_mask) + epsilon

        # Calcular a média apenas nesses pixels
        leaf_mse = paddle.sum(leaf_error_sq) / leaf_pixel_count
        square_mse = paddle.sum(square_error_sq) / square_pixel_count

        area_loss = leaf_mse + square_mse
    
        return [coef_ce * seg_loss, coef_mse * area_loss]
    
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
        # conv2D
        # dropout
        x = self.conv_seg(x)  # Logits por classe (B, num_classes, h, w)
        return x
    