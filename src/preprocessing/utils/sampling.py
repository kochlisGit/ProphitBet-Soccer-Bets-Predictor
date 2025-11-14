import numpy as np
from enum import Enum
from typing import Optional, Tuple, Union
from imblearn.base import BaseSampler
from imblearn.over_sampling import SVMSMOTE
from imblearn.under_sampling import NearMiss, InstanceHardnessThreshold


class SamplerType(Enum):
    """ The supported sampling methods. """

    NEARMISS = 'nearmiss'
    INSTANCE_HARDNESS_THRESHOLD = 'instance-hardness-threshold'
    SVM_SMOTE = 'svm-smote'


def get_sampler(sampler: SamplerType, seed: Optional[int] = None) -> BaseSampler:
    """ Sampler method factory. """

    if sampler == SamplerType.NEARMISS:
        return NearMiss(version=3, n_neighbors_ver3=5, n_neighbors=5, n_jobs=-1)
    elif sampler == SamplerType.INSTANCE_HARDNESS_THRESHOLD:
        return InstanceHardnessThreshold(random_state=seed, n_jobs=-1)
    elif sampler == SamplerType.SVM_SMOTE:
        return SVMSMOTE(random_state=seed)
    else:
        raise NotImplementedError(f'Not supported sampler type: "{sampler.name}"')


def sample(
        x: np.ndarray,
        y: np.ndarray,
        sampler: Union[SamplerType, BaseSampler],
        seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, BaseSampler]:
    """ Samples from the input dataframe and returns the sampled dataframe along with the sampler instance. """

    if isinstance(sampler, SamplerType):
        sampler = get_sampler(sampler=sampler, seed=seed)

    x_sampled, y_sampled = sampler.fit_resample(x, y)
    return x_sampled, y_sampled, sampler
