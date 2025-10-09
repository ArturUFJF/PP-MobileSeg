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
    
    #o novo decoder deve:
    # 1) Calcular o tamanho real de cada pixel da classe leaf calculando a razão entre área real da folha e total de pixels da folha na máscara binária, extraindo informações dos xml
    # 2) Calcular o tamanho real de cada pixel da classe square calculando a razão entre área real do quadrado e total de pixels da máscara binária do quadrado, levando em conta a estimativa de pose, extraindo informações dos xml
    # 3) Retornar os logits com cada área calculada para cada pixel das folhas e quadrados
    # 4) Realizar o Produto de Hadamard entre os logits e as máscaras binárias para obter as estimativas de área finais para cada classe

    #A nova loss function deve:
    # - Ser a soma da loss function MSE dos 2 decoders