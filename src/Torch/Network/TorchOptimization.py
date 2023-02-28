from typing import Dict, Any
from torch import Tensor, reshape, save, load
from collections import namedtuple

from DeepPhysX.Core.Network.BaseOptimization import BaseOptimization
from DeepPhysX.Torch.Network.TorchNetwork import TorchNetwork


class TorchOptimization(BaseOptimization):

    def __init__(self,
                 config: namedtuple):
        """
        TorchOptimization computes loss between prediction and target and optimizes the Network parameters.

        :param config: Set of TorchOptimization parameters.
        """

        BaseOptimization.__init__(self, config)

    def set_loss(self) -> None:
        """
        Initialize the loss function.
        """

        if self.loss_class is not None:
            self.loss = self.loss_class()

    def compute_loss(self,
                     data_pred: Dict[str, Tensor],
                     data_opt: Dict[str, Tensor]) -> Dict[str, Any]:
        """
        Compute loss from prediction / ground truth.

        :param data_pred: Tensor produced by the forward pass of the Network.
        :param data_opt: Ground truth tensor to be compared with prediction.
        :return: Loss value.
        """

        self.loss_value = self.loss(data_pred['prediction'].view(data_opt['ground_truth'].shape),
                                    data_opt['ground_truth'])
        return self.transform_loss(data_opt)

    def transform_loss(self,
                       data_opt: Dict[str, Tensor]) -> Dict[str, float]:
        """
        Apply a transformation on the loss value using the potential additional data.

        :param data_opt: Additional data sent as dict to compute loss value.
        :return: Transformed loss value.
        """

        return {'loss': self.loss_value.item()}

    def set_optimizer(self,
                      net: TorchNetwork) -> None:
        """
        Define an optimization process.

        :param net: Network whose parameters will be optimized.
        """

        if (self.optimizer_class is not None) and (self.lr is not None):
            self.optimizer = self.optimizer_class(net.parameters(), self.lr)

    def optimize(self) -> None:
        """
        Run an optimization step.
        """

        self.optimizer.zero_grad()
        self.loss_value.backward()
        self.optimizer.step()

    def load_parameters(self,
                        path: str,
                        device: any = None) -> None:
        """
        Load network parameter from path.

        :param path: Path to Network parameters to load.
        """

        self.optimizer.load_state_dict(load(path, map_location=device))
        # Update the lr to the current lr
        for g in self.optimizer.param_groups:
            g['lr'] = self.lr

    def save_parameters(self,
                        path: str) -> None:
        """
        Saves the optimizer parameters to the path location.

        :param path: Path where to save the parameters.
        """

        path = path + '.pth'
        save(self.optimizer.state_dict(), path)


    def __str__(self) -> str:

        return BaseOptimization.__str__(self)
