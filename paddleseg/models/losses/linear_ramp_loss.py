import paddle
import paddle.nn as nn
from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class LinearRampLoss(nn.Layer):
    def __init__(self,
                 start_coefs=[1.0, 0.0],
                 end_coefs=[0.5, 0.5],
                 total_iters=10000,
                 ignore_index=255):
        super().__init__()
        if len(start_coefs) != 2 or len(end_coefs) != 2:
            raise ValueError('`start_coefs` and `end_coefs` must have length 2.')
        if total_iters <= 0:
            raise ValueError('`total_iters` must be > 0.')

        self.cross_entropy = manager.LOSSES['CrossEntropyLoss'](
            ignore_index=ignore_index)
        self.lovasz = manager.LOSSES['LovaszSoftmaxLoss'](
            ignore_index=ignore_index)
        self.start_coefs = start_coefs
        self.end_coefs = end_coefs
        self.total_iters = total_iters
        self.current_iter = 0

    def forward(self, logits, label):
        # Calcula o fator de progresso (0 a 1)
        alpha = min(self.current_iter / self.total_iters, 1.0)
        
        # Interpolação linear dos coeficientes
        c1 = self.start_coefs[0] + alpha * (self.end_coefs[0] - self.start_coefs[0])
        c2 = self.start_coefs[1] + alpha * (self.end_coefs[1] - self.start_coefs[1])
        
        loss_ce = self.cross_entropy(logits, label)
        loss_lovasz = self.lovasz(logits, label)
        
        if not self.training:  # During eval, keep final configured weighting.
            return self.end_coefs[0] * loss_ce + self.end_coefs[1] * loss_lovasz

        self.current_iter += 1
        return c1 * loss_ce + c2 * loss_lovasz