import paddle
from paddle import nn

from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class L2Loss(nn.Layer):
    """Squared L2 loss with optional ignore-index masking.

    With ``reduction='none'``, returns ``(input - label) ** 2`` elementwise.
    """

    def __init__(self, reduction='mean', ignore_index=255):
        super().__init__()
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(
                "The value of 'reduction' in L2Loss should be 'sum', 'mean' or 'none', "
                "but received %s, which is not allowed." % reduction)
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.EPS = 1e-10

    def forward(self, input, label):
        mask = paddle.cast(label != self.ignore_index, input.dtype)
        label.stop_gradient = True
        mask.stop_gradient = True

        output = paddle.square(input - label) * mask

        if self.reduction == 'mean':
            return paddle.sum(output) / (paddle.sum(mask) + self.EPS)
        if self.reduction == 'none':
            return output
        return paddle.sum(output)
