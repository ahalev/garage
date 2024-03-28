"""Interface of RLAlgorithm."""
import abc

from torch import nn
from garage.torch import as_torch_dict, global_device, state_dict_to


class RLAlgorithm(abc.ABC):
    """Base class for all the algorithms.

    Note:
        If the field sampler_cls exists, it will be by Trainer.setup to
        initialize a sampler.

    """

    # pylint: disable=too-few-public-methods

    @abc.abstractmethod
    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Trainer is passed to give algorithm
                the access to trainer.step_epochs(), which provides services
                such as snapshotting and sampler control.

        """

    @property
    def networks(self):
        """
        Return torch networks.
        """
        return [attr for attr in self.__dict__.values() if isinstance(attr, nn.Module)]

    def to(self, device=None):
        if device is None:
            device = global_device()

        for net in self.networks:
            net.to(device)
