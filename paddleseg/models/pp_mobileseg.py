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
    """
    logits_list: list de logits produzidos pelo forward (espera-se [seg_logits, area_logits])
    losses: dict contendo 'types' (lista de loss objects) e 'coef' (lista de coeficientes)
    data: dicionário do dataset contendo pelo menos 'label' e 'areaLabel'
    Retorna: lista de perdas ponderadas (cada item já multiplicado pelo coef)
    """
    # Garantir a consistência da entrada
    if not isinstance(logits_list, (list, tuple)):
        raise RuntimeError("logits_list deve ser list/tuple com duas entradas (seg, area)")

    if len(logits_list) != len(losses['types']):
        # A checagem do train.py também verifica, mas é útil explicitar
        raise RuntimeError(f"Comprimento de logits_list ({len(logits_list)}) != número de loss types ({len(losses['types'])})")

    # Extrair rótulos do dicionário (e converter para int64, remover eixo channel singleton se precisar)
    labels = data.get('label', None)
    area_labels = data.get('areaLabel', None)

    if labels is None or area_labels is None:
        raise RuntimeError("data deve conter 'label' e 'areaLabel' para calcular as perdas")

    # squeeze se tiver shape (N,1,H,W)
    if labels.ndim == 4 and labels.shape[1] == 1:
        labels = paddle.squeeze(labels, axis=1)
    if area_labels.ndim == 4 and area_labels.shape[1] == 1:
        area_labels = paddle.squeeze(area_labels, axis=1)

    labels = labels.astype('int64')
    area_labels = area_labels.astype('int64')

    loss_list = []
    # para cada logit, calcular a loss correspondente.
    # espera-se que losses['types'] esteja alinhado com a ordem de logits_list.
    for i, logits in enumerate(logits_list):
        loss_fn = losses['types'][i]
        coef = losses['coef'][i] if 'coef' in losses and len(losses['coef']) > i else 1.0

        # Assumimos: index 0 -> segmentation (usa labels), index 1 -> area (usa area_labels)
        if i == 0:
            tgt = labels
        elif i == 1:
            tgt = area_labels
        else:
            # caso você tenha mais cabeças, por enquanto usamos labels por padrão
            tgt = labels

        # Algumas losses específicas podem esperar edges ou outro formato; aqui tratamos o caso comum:
        # loss_fn espera (logits, labels)
        loss_val = loss_fn(logits, tgt)
        loss_list.append(coef * loss_val)

    return loss_list