from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from models.model import ScikitModel


class NaiveBayes(ScikitModel):
    def __init__(
            self,
            model_id: str,
            algorithm: str = 'gaussian',
            calibrate_probabilities: bool = False,
            **kwargs
    ):
        if algorithm == 'gaussian':
            self._estimator_cls = GaussianNB
        elif algorithm == 'multinomial':
            self._estimator_cls = MultinomialNB
        else:
            assert algorithm == 'complement', f'Not supported algorithm: "{algorithm}"'

            self._estimator_cls = ComplementNB

        super().__init__(
            model_id=model_id,
            model_name='naive-bayes',
            calibrate_probabilities=calibrate_probabilities,
            **kwargs
        )

    def _build_estimator(self, input_size: int, num_classes: int) -> GaussianNB or MultinomialNB or ComplementNB:
        return self._estimator_cls()
