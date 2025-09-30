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
        x = self.decode_head(x)  # Converte features em logits por classe (B, num_classes, h, w)

        # Estratégia de upsample:
        # - Durante treino (self.training == True), sempre usa interpolate bilinear.
        # - Também usa interpolate se upsample == 'intepolate' (padrão) ou num_classes < 30 (heurística).
        if self.upsample == 'intepolate' or self.training or self.num_classes < 30:
            x = F.interpolate(
                x, x_hw, mode='bilinear', align_corners=self.align_corners)
        elif self.upsample == 'vim':
            # Modo VIM: otimização para reduzir custo de memória/cálculo na interpolação
            # 1) Obtém o conjunto de rótulos presentes no mapa de predição de baixa resolução
            labelset = paddle.unique(paddle.argmax(x, 1))  # shape: (K,), K = nº de classes presentes
            # 2) Mantém apenas os canais das classes presentes (reduzindo de C para K)
            x = paddle.gather(x, labelset, axis=1)
            # 3) Faz o upsample apenas nesses K canais
            x = F.interpolate(
                x, x_hw, mode='bilinear', align_corners=self.align_corners)

            # 4) Reconstroi as classes originais: argmax nos K canais e remapeia para índices verdadeiros
            pred = paddle.argmax(x, 1)  # mapa (B, H, W) com índices 0..K-1
            pred_retrieve = paddle.zeros(pred.shape, dtype='int32')
            for i, val in enumerate(labelset):
                # Para cada índice reduzido i, mapeia de volta para o rótulo global labelset[i]
                pred_retrieve[pred == i] = labelset[i].cast('int32')

            # No modo VIM, a saída final é o mapa de rótulos inteiros (sem logits)
            x = pred_retrieve
        else:
            # Caso seja passado um modo de upsample não implementado
            raise NotImplementedError(self.upsample, " is not implemented")

        # Retorno como lista, seguindo a convenção do PaddleSeg (permite múltiplas saídas)
        return [x]


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
    
    #Solução sugerida para adicionar cálculo de áreas por classe, em análise e adaptação
#     # ...existing code...
# class PPMobileSeg(nn.Layer):
#     """
#     Implementação do PP_MobileSeg baseada em PaddlePaddle.
#     Referência: https://arxiv.org/abs/2304.05152

#     Args:
#         num_classes (int): Número de classes alvo.
#         backbone (nn.Layer): Backbone de extração de features (deve definir feat_channels).
#         head_use_dw (bool, opcional): Se a cabeça usa convoluções "depthwise" (via groups).
#         align_corners (bool, opcional): Parâmetro do resize bilinear para alinhamento de cantos.
#         pretrained (str, opcional): Caminho/URL de pesos pré-treinados para carregar no modelo.
#         upsample (str, opcional): Tipo de upsample. 'intepolate' (padrão) ou 'vim' para otimização.
#                                  Obs.: 'intepolate' aqui é uma grafia mantida pelo código de origem.
# +       return_areas (bool, opcional): Se True, também retorna áreas por classe (B, num_classes).
# +       area_mode (str, opcional): 'argmax' para contagem discreta de pixels, 'soft' para soma das probabilidades.
#     """
# # ...existing code...
#     def __init__(self,
#                  num_classes,
#                  backbone,
#                  head_use_dw=True,
#                  align_corners=False,
#                  pretrained=None,
# -                upsample='intepolate'):
# +                upsample='intepolate',
# +                return_areas=False,
# +                area_mode='argmax'):
#         super().__init__()
#         # Guarda referências e hiperparâmetros
#         self.backbone = backbone  # Backbone deve retornar um mapa de features compatível com a cabeça
#         self.upsample = upsample  # Modo de upsample ('intepolate' padrão ou 'vim' otimizado)
#         self.num_classes = num_classes
# +        self.return_areas = return_areas
# +        self.area_mode = area_mode  # 'argmax' | 'soft'
# # ...existing code...
#     def forward(self, x):
#         # x: tensor de entrada (B, C, H, W)
#         x_hw = x.shape[2:]  # Guarda a resolução original para upsample posterior
#         x = self.backbone(x)  # Extrai features com o backbone
#         x = self.decode_head(x)  # Converte features em logits por classe (B, num_classes, h, w)

#         # Estratégia de upsample:
#         # - Durante treino (self.training == True), sempre usa interpolate bilinear.
#         # - Também usa interpolate se upsample == 'intepolate' (padrão) ou num_classes < 30 (heurística).
#         if self.upsample == 'intepolate' or self.training or self.num_classes < 30:
#             x = F.interpolate(
#                 x, x_hw, mode='bilinear', align_corners=self.align_corners)
#         elif self.upsample == 'vim':
#             # Modo VIM: otimização para reduzir custo de memória/cálculo na interpolação
#             # 1) Obtém o conjunto de rótulos presentes no mapa de predição de baixa resolução
#             labelset = paddle.unique(paddle.argmax(x, 1))  # shape: (K,), K = nº de classes presentes
#             # 2) Mantém apenas os canais das classes presentes (reduzindo de C para K)
#             x = paddle.gather(x, labelset, axis=1)
#             # 3) Faz o upsample apenas nesses K canais
#             x = F.interpolate(
#                 x, x_hw, mode='bilinear', align_corners=self.align_corners)

#             # 4) Reconstroi as classes originais: argmax nos K canais e remapeia para índices verdadeiros
#             pred = paddle.argmax(x, 1)  # mapa (B, H, W) com índices 0..K-1
#             pred_retrieve = paddle.zeros(pred.shape, dtype='int32')
#             for i, val in enumerate(labelset):
#                 # Para cada índice reduzido i, mapeia de volta para o rótulo global labelset[i]
#                 pred_retrieve[pred == i] = labelset[i].cast('int32')

#             # No modo VIM, a saída final é o mapa de rótulos inteiros (sem logits)
#             x = pred_retrieve
#         else:
#             # Caso seja passado um modo de upsample não implementado
#             raise NotImplementedError(self.upsample, " is not implemented")

# -        # Retorno como lista, seguindo a convenção do PaddleSeg (permite múltiplas saídas)
# -        return [x]
# +        # Opcional: calcular áreas por classe na resolução final.
# +        if self.return_areas:
# +            areas = self._compute_areas(x)
# +            return [x, areas]  # x: logits (float) ou mapa (int) no modo 'vim'; areas: (B, num_classes)
# +        else:
# +            # Retorno como lista, seguindo a convenção do PaddleSeg (permite múltiplas saídas)
# +            return [x]
# +
# +    def _compute_areas(self, x):
# +        """
# +        Calcula áreas por classe (número de pixels) para cada imagem do batch.
# +        - Se x for logits (float, shape [B, C, H, W]): usa area_mode:
# +            - 'argmax': conta pixels do rótulo vencedor.
# +            - 'soft'   : soma probabilidades (Softmax) por classe ao longo de HxW.
# +        - Se x for mapa de rótulos (int, shape [B, H, W]) do modo 'vim': conta pixels por classe.
# +        Retorna tensor shape [B, num_classes] com dtype float32 (soft) ou int64 (argmax/labelmap).
# +        """
# +        # Caso x seja mapa de rótulos (modo 'vim'): int tensor [B, H, W]
# +        if x.dtype in [paddle.int32, paddle.int64] and len(x.shape) == 3:
# +            one_hot = F.one_hot(x, num_classes=self.num_classes)  # [B, H, W, C]
# +            areas = paddle.sum(one_hot, axis=[1, 2])  # [B, C]
# +            return areas.cast('int64')
# +
# +        # Caso x sejam logits: float tensor [B, C, H, W]
# +        if self.area_mode == 'soft':
# +            probs = F.softmax(x, axis=1)             # [B, C, H, W]
# +            areas = paddle.sum(probs, axis=[2, 3])   # [B, C] soma das probabilidades
# +            return areas  # float32
# +        else:
# +            pred = paddle.argmax(x, axis=1)          # [B, H, W]
# +            one_hot = F.one_hot(pred, num_classes=self.num_classes)  # [B, H, W, C]
# +            areas = paddle.sum(one_hot, axis=[1, 2])                 # [B, C]
# +            return areas.cast('int64')
# # ...existing code...