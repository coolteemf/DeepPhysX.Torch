import inspect
from typing import Dict, Any
from torch import Tensor, reshape, save, load
from torch.optim.lr_scheduler import CyclicLR
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
        self.optimizer_parameters = config.optimizer_parameters if hasattr(config, 'optimizer_parameters') else None
        self.loss_parameters = config.loss_parameters if hasattr(config, 'loss_parameters') else None
        self.scheduler_class = config.scheduler_class if hasattr(config, 'scheduler_class') else None
        self.scheduler_parameters = config.scheduler_parameters if hasattr(config, 'scheduler_parameters') else None
        self.scheduler = None
        self.schedule_on_epoch = True
        self.schedule_on_batch = False
        if self.scheduler_class is CyclicLR:
            self.schedule_on_epoch = False
            self.schedule_on_batch = True

    def schedule_lr(self, *args, **kwargs):
        """
        Adjusts the lr if scheduler is not None. Should be called after train and validate.
        """
        if self.scheduler is not None:
            self.scheduler.step(*args, **kwargs)
            self.lr = self.optimizer.param_groups[0]['lr']

    def set_loss(self) -> None:
        """
        Initialize the loss function.
        """

        if self.loss_class is not None:
            if self.loss_parameters is not None:
                if 'device' in inspect.signature(self.loss_class.__init__).parameters:
                    self.loss_parameters.update({'device': self.manager.network.device})
            self.loss = self.loss_class(**({} if self.loss_parameters is None else self.loss_parameters))

    def compute_loss(self,
                     data_pred: Dict[str, Tensor],
                     data_opt: Dict[str, Tensor]) -> Dict[str, Any]:
        """
        Compute loss from prediction / ground truth.

        :param data_pred: Tensor produced by the forward pass of the Network.
        :param data_opt: Ground truth tensor to be compared with prediction.
        :return: Loss value.
        """
        self.loss_value = self.loss(data_pred, data_opt)
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
        | Define an optimization process.
        | Set the optimizer object with parameters.
        | Set the lr scheduler with parameters.

        :param net: Network whose parameters will be optimized.
        """

        if (self.optimizer_class is not None) and (self.lr is not None):
            self.optimizer = self.optimizer_class(net.parameters(), self.lr,
                                                  **({} if self.optimizer_parameters is None else
                                                     self.optimizer_parameters))
            if self.scheduler_class is not None:
                self.scheduler = self.scheduler_class(self.optimizer,
                                                      **({} if self.scheduler_parameters is None else
                                                         self.scheduler_parameters))

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
        loaded = load(path, map_location=device)
        self.optimizer.load_state_dict(loaded)
        # Update the lr to the current lr
        for g in self.optimizer.param_groups:
            g['lr'] = self.lr
            for k, v in ({} if self.optimizer_parameters is None else self.optimizer_parameters).items():
                g[k] = v

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
