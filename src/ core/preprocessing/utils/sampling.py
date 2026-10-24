import numpy as np
from enum import Enum
from typing import Optional, Tuple, Union
from imblearn.base import BaseSampler
from imblearn.over_sampling import SVMSMOTE
from imblearn.under_sampling import NearMiss, InstanceHardnessThreshold


class SamplerType(Enum):
    """ The supported sampling methods.

        Supported Types:
            1. Near-Miss (Undersampling Method).
            2. Instance Hardness Threshold (Undersampling Method).
            3. SVM-SMOTE (Oversampling Method).
    """

    NEARMISS = 'Near-Miss'
    INSTANCE_HARDNESS_THRESHOLD = 'Instance-Hardness-Threshold'
    SVM_SMOTE = 'SVM-Smote'

    @classmethod
    def from_str(cls, value: str) -> "SamplerType":
        """ Converts string labels or names into SamplerType.

            Args
            ----
            value: string value of the sampling method.
        """

        try:
            return cls(value)
        except ValueError:
            raise ValueError(f'Not supported sampler type: "{value}".')

    def create(self, seed: Optional[int] = 0) -> BaseSampler:
        """ Factory method that creates a new sampler instance. """

        if self == SamplerType.NEARMISS:
            return NearMiss(version=3, n_neighbors_ver3=5, n_neighbors=5, n_jobs=-1)
        elif self == SamplerType.INSTANCE_HARDNESS_THRESHOLD:
            return InstanceHardnessThreshold(random_state=seed, n_jobs=-1)
        elif self == SamplerType.SVM_SMOTE:
            return SVMSMOTE(random_state=seed)
        else:
            raise NotImplementedError(f'Not supported sampler type: "{self.name}"')


def sample_inputs(
        x: np.ndarray,
        y: np.ndarray,
        sampler: Union[str, SamplerType, BaseSampler],
        seed: Optional[int] = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """ Samples from the input dataframe and returns the sampled inputs and targets.

        Args
        ----
        :param x: (np.ndarray) input array.
        :param y: (np.ndarray) target array.
        :param sampler: (str | SamplerType | BaseSampler]) specified sampler (type or instance).
        :param seed: (Optional[int]) random seed for the sampler. Defaults to None.
    """

    if isinstance(sampler, str):
        sampler = SamplerType.from_str(sampler)
    if isinstance(sampler, SamplerType):
        sampler = sampler.create(seed=seed)

    x_sampled, y_sampled = sampler.fit_resample(x, y)
    return x_sampled, y_sampled
