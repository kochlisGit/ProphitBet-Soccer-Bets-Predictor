import numpy as np
import pandas as pd
import shap
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple
from shap.maskers import Independent
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from src.core.database.model import ModelDatabase
from src.core.interpretability.explainer import ClassifierExplainer
from src.core.preprocessing.dataset import construct_targets

_BACKGROUND_IMAGE_SIZE = 500
_BOUNDARY_GRID_SIZE = 180
_GAUSSIAN_POINTS = 700
_ELLIPSE_POINTS = 220
_GAUSSIAN_N_STD = 2.0
_BACKGROUND_MAX_STRENGTH = 0.58
_PREDICTION_BATCH_SIZE = 50_000
_CLASS_RGB = np.asarray([[220, 65, 65], [55, 135, 205], [55, 165, 95],], dtype=np.float64)


@dataclass(frozen=True)
class DiscriminantPlotData:
    """ Fully prepared numeric data for a transformed-space DA plot.

        Expensive model inference and geometry construction are deliberately done
        before this object is returned. The GUI only needs to turn these arrays
        into Qt chart series and an image.
    """

    title: str
    decision_boundary: str
    class_names: Tuple[str, ...]
    class_values: np.ndarray
    y_true: np.ndarray
    z: np.ndarray
    sample_probabilities: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    background_rgba: np.ndarray
    gaussian_x: np.ndarray
    gaussian_pdfs: np.ndarray
    boundary_positions: np.ndarray
    ellipses: Tuple[np.ndarray, ...]
    boundary_polylines: Tuple[np.ndarray, ...]

    @property
    def is_1d(self) -> bool:
        return self.z.shape[1] == 1

    @property
    def n_classes(self) -> int:
        return len(self.class_names)


def _probability_background_rgba(probabilities: np.ndarray) -> np.ndarray:
    """Convert posterior probabilities to the white-to-class-color backdrop."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError('Probability background expects a two-dimensional array with at least two classes.')

    n_classes = probabilities.shape[1]
    if n_classes > len(_CLASS_RGB):
        raise ValueError(f'The discriminant plot supports at most {len(_CLASS_RGB)} classes, got {n_classes}.')

    predicted = np.argmax(probabilities, axis=1)
    confidence = probabilities[np.arange(len(probabilities)), predicted]
    chance = 1.0 / n_classes
    normalized = np.clip((confidence - chance) / max(1e-12, 1.0 - chance), 0.0, 1.0)
    strength = _BACKGROUND_MAX_STRENGTH * np.power(normalized, 0.72)
    base_rgb = _CLASS_RGB[predicted]
    rgb = 255.0 * (1.0 - strength[:, None]) + base_rgb * strength[:, None]
    rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    alpha = np.full((len(rgb), 1), 255, dtype=np.uint8)
    return np.concatenate((rgb, alpha), axis=1)


def _positive_semidefinite_covariance(covariance: np.ndarray, dimensions: int) -> np.ndarray:
    """Return a finite symmetric covariance with tiny eigenvalues clipped."""

    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.size != dimensions * dimensions:
        covariance = np.eye(dimensions, dtype=np.float64) * 1e-12
    else:
        covariance = covariance.reshape(dimensions, dimensions)
        covariance = np.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
        covariance = 0.5 * (covariance + covariance.T)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def _covariance_ellipse(
        mean: np.ndarray,
        covariance: np.ndarray,
        n_std: float = _GAUSSIAN_N_STD,
        n_points: int = _ELLIPSE_POINTS,
) -> np.ndarray:
    covariance = _positive_semidefinite_covariance(covariance, dimensions=2)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    theta = np.linspace(0.0, 2.0 * np.pi, n_points)
    unit_circle = np.vstack((np.cos(theta), np.sin(theta)))
    radii = n_std * np.sqrt(eigenvalues)
    ellipse = eigenvectors @ (radii[:, None] * unit_circle)
    return ellipse.T + np.asarray(mean, dtype=np.float64).reshape(1, 2)


def _gaussian_pdf_1d(x: np.ndarray, mean: float, variance: float) -> np.ndarray:
    variance = max(float(variance), 1e-12)
    standard_deviation = np.sqrt(variance)
    return np.exp(-0.5 * ((x - mean) / standard_deviation) ** 2) / (standard_deviation * np.sqrt(2.0 * np.pi))


def _interpolate_zero_crossing(
        first_point: tuple[float, float],
        first_value: float,
        second_point: tuple[float, float],
        second_value: float,
) -> tuple[float, float]:
    denominator = first_value - second_value
    interpolation = 0.5 if abs(denominator) < 1e-14 else first_value / denominator
    interpolation = float(np.clip(interpolation, 0.0, 1.0))
    return (
        first_point[0] + interpolation * (second_point[0] - first_point[0]),
        first_point[1] + interpolation * (second_point[1] - first_point[1]),
    )


def _marching_squares_segments(
        x_values: np.ndarray,
        y_values: np.ndarray,
        scalar_field: np.ndarray,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Extract zero-contour line segments from a scalar grid."""

    segments = []
    rows, columns = scalar_field.shape

    for row in range(rows - 1):
        y0 = float(y_values[row])
        y1 = float(y_values[row + 1])

        for column in range(columns - 1):
            x0 = float(x_values[column])
            x1 = float(x_values[column + 1])
            p00, p10 = (x0, y0), (x1, y0)
            p11, p01 = (x1, y1), (x0, y1)
            v00 = float(scalar_field[row, column])
            v10 = float(scalar_field[row, column + 1])
            v11 = float(scalar_field[row + 1, column + 1])
            v01 = float(scalar_field[row + 1, column])

            crossings = []
            for point_a, value_a, point_b, value_b in (
                    (p00, v00, p10, v10),
                    (p10, v10, p11, v11),
                    (p11, v11, p01, v01),
                    (p01, v01, p00, v00),
            ):
                if value_a == 0.0 and value_b == 0.0:
                    continue
                if (value_a <= 0.0 < value_b) or (value_b <= 0.0 < value_a):
                    crossings.append(
                        _interpolate_zero_crossing(point_a, value_a, point_b, value_b)
                    )

            unique_crossings = []
            for point in crossings:
                if not any(
                        abs(point[0] - other[0]) < 1e-10
                        and abs(point[1] - other[1]) < 1e-10
                        for other in unique_crossings
                ):
                    unique_crossings.append(point)

            if len(unique_crossings) == 2:
                segments.append((unique_crossings[0], unique_crossings[1]))
            elif len(unique_crossings) == 4:
                # Match the reference plot's deterministic saddle-cell pairing.
                segments.append((unique_crossings[0], unique_crossings[1]))
                segments.append((unique_crossings[2], unique_crossings[3]))

    return segments


def _connect_segments(
        segments: Sequence[tuple[tuple[float, float], tuple[float, float]]],
        decimals: int = 7,
) -> tuple[np.ndarray, ...]:
    """Connect adjacent marching-squares segments into chart-ready polylines."""

    if not segments:
        return ()

    def key(point: tuple[float, float]) -> tuple[float, float]:
        return round(point[0], decimals), round(point[1], decimals)

    adjacency = {}
    for segment_id, segment in enumerate(segments):
        for endpoint_id, point in enumerate(segment):
            adjacency.setdefault(key(point), []).append((segment_id, endpoint_id))

    unused = set(range(len(segments)))
    polylines = []
    while unused:
        segment_id = unused.pop()
        start, end = segments[segment_id]
        polyline = [start, end]

        while True:
            next_segment_id = None
            next_point = None
            for candidate_id, endpoint_id in adjacency.get(key(polyline[-1]), []):
                if candidate_id in unused:
                    candidate = segments[candidate_id]
                    next_segment_id = candidate_id
                    next_point = candidate[1] if endpoint_id == 0 else candidate[0]
                    break
            if next_segment_id is None:
                break
            unused.remove(next_segment_id)
            polyline.append(next_point)

        while True:
            next_segment_id = None
            next_point = None
            for candidate_id, endpoint_id in adjacency.get(key(polyline[0]), []):
                if candidate_id in unused:
                    candidate = segments[candidate_id]
                    next_segment_id = candidate_id
                    next_point = candidate[1] if endpoint_id == 0 else candidate[0]
                    break
            if next_segment_id is None:
                break
            unused.remove(next_segment_id)
            polyline.insert(0, next_point)

        polylines.append(np.asarray(polyline, dtype=np.float64))

    return tuple(polylines)


class DiscriminantAnalysisExplainer(ClassifierExplainer):
    """Explainer for linear and quadratic discriminant-analysis models."""

    def __init__(self, df: pd.DataFrame, model_id: str, database: ModelDatabase):
        super().__init__(df=df, model_id=model_id, database=database)
        self._decision_boundary = self.model_config.classifier_config['decision_boundary']

    def _compute_shap_values(self, df: pd.DataFrame) -> shap.Explanation:
        clf = self.model.classifier

        if self.model_config.classifier_config['calibrate_probabilities']:
            clf = clf.calibrated_classifiers_[0].estimator

        if self._decision_boundary == 'linear':
            explainer = shap.LinearExplainer(model=clf, masker=Independent(df, max_samples=df.shape[0]))
            return explainer(df)

        def predict_probabilities(values) -> np.ndarray:
            # PermutationExplainer requires a callable model. Converting to an
            # ndarray also avoids sklearn feature-name warnings because the
            # stored estimator was fitted on preprocessed arrays.
            return clf.predict_proba(np.asarray(values))

        masker = Independent(df, max_samples=min(df.shape[0], 100))
        explainer = shap.PermutationExplainer(model=predict_probabilities, masker=masker, feature_names=df.columns.tolist())
        return explainer(df, max_evals=2 * df.shape[1] + 1)

    def get_coefficients(self) -> Optional[pd.DataFrame]:
        if self._decision_boundary != 'linear':
            return None

        coefficients = self.model.get_coefficients()

        if len(self.target_names) < 3:
            return pd.DataFrame({'Feature': self.feature_names, 'Coefficient': coefficients})
        else:
            return pd.DataFrame(data=coefficients, columns=self.feature_names, index=self.target_names)

    def get_transformed_plot_data(self) -> DiscriminantPlotData:
        """ Prepare all numeric work required by the transformed-space plot.

            A visualization-only SVD LDA supplies a stable one- or two-dimensional
            projection for regular LDA and QDA. OAS LDA uses its fitted
            coefficient subspace because sklearn's ``lsqr`` solver has no
            ``transform`` method. Posterior values are always produced by the
            actual stored classifier: projected query points are mapped back to
            the retained input subspace before ``predict_proba`` is called.

            This method intentionally includes the expensive background grids,
            boundary extraction, Gaussian curves, and ellipse geometry so callers
            can execute it in their existing explanation worker task.
        """

        x_train, _ = self._dataset_preprocessor.construct_inputs(df=self.train_df, normalizer=self.normalizer,)
        y_train = construct_targets(df=self.train_df, target_type=self.target_type, one_hot=False)
        x_train = np.asarray(x_train, dtype=np.float64)
        y_train = np.asarray(y_train).reshape(-1)

        if x_train.ndim != 2 or x_train.shape[0] == 0 or x_train.shape[1] == 0:
            raise ValueError('The transformed-space plot requires a non-empty two-dimensional input array.')
        if len(y_train) != len(x_train):
            raise ValueError(
                f'Input and target sample counts differ: {len(x_train)} and {len(y_train)}.'
            )
        if not np.isfinite(x_train).all():
            raise ValueError('The transformed-space plot cannot be built from non-finite inputs.')

        target_mapper = self.target_type.get_mapper()
        class_names = tuple(str(name) for name in target_mapper.keys())
        class_values = np.asarray(list(target_mapper.values()))
        observed_values = np.unique(y_train)
        missing_values = [value for value in class_values if value not in observed_values]
        if missing_values:
            missing_names = [
                class_names[index]
                for index, value in enumerate(class_values)
                if value in missing_values
            ]
            raise ValueError(
                f'The training split does not contain every configured target class. Missing: {", ".join(missing_names)}.'
            )
        if len(class_values) < 2:
            raise ValueError('The transformed-space plot requires at least two target classes.')
        if len(class_values) > len(_CLASS_RGB):
            raise ValueError(f'The transformed-space plot supports at most {len(_CLASS_RGB)} classes.')

        n_components = min(len(class_values) - 1, x_train.shape[1])
        projection = LinearDiscriminantAnalysis(solver='svd', n_components=n_components, store_covariance=True)
        fallback_z = np.asarray(projection.fit_transform(x_train, y_train), dtype=np.float64)
        fallback_scalings = np.asarray(projection.scalings_[:, :n_components], dtype=np.float64)
        classifier = self.model.classifier
        use_fitted_oas_space = self._decision_boundary == 'linear' and bool(self.model_config.classifier_config.get('oas', False))

        if use_fitted_oas_space:
            # ``lsqr`` LDA has no transform(). Its fitted coefficient-difference
            # subspace is nevertheless exactly the discriminant subspace that
            # controls uncalibrated posterior probabilities. Build the visual
            # coordinates from that space instead of from a separately fitted,
            # non-shrunk SVD LDA.
            scalings = self._coefficient_projection_basis(
                classifier=classifier,
                fallback_scalings=fallback_scalings,
                n_components=n_components,
            )
            center = np.mean(x_train, axis=0, dtype=np.float64)
            z = (x_train - center) @ scalings
        else:
            scalings = fallback_scalings
            center = np.asarray(projection.xbar_, dtype=np.float64)
            z = fallback_z

        if z.ndim == 1:
            z = z[:, None]

        dimensions = z.shape[1]
        if dimensions not in (1, 2):
            # The application's target types currently produce at most two
            # components; retaining this check makes the rendering contract
            # explicit if another target type is added later.
            raise ValueError(f'The transformed-space plot supports one or two dimensions, got {dimensions}.')

        inverse_scalings = np.linalg.pinv(scalings)
        model_classes = np.asarray(getattr(classifier, 'classes_', class_values))

        missing_model_values = [value for value in class_values if value not in model_classes]
        if missing_model_values:
            raise ValueError(
                'The stored classifier does not expose probabilities for every configured target class.'
            )

        class_column_indices = np.asarray(
            [int(np.flatnonzero(model_classes == value)[0]) for value in class_values],
            dtype=np.int64,
        )

        def projected_probabilities(query: np.ndarray) -> np.ndarray:
            query = np.asarray(query, dtype=np.float64)
            query = np.atleast_2d(query)
            if query.shape[1] != dimensions:
                raise ValueError(
                    f'Projected queries must have {dimensions} columns, got {query.shape[1]}.'
                )

            probability_batches = []
            for start in range(0, len(query), _PREDICTION_BATCH_SIZE):
                batch = query[start:start + _PREDICTION_BATCH_SIZE]
                reconstructed = batch @ inverse_scalings + center
                raw_probabilities = np.asarray(
                    classifier.predict_proba(reconstructed),
                    dtype=np.float64,
                )
                if raw_probabilities.ndim != 2 or raw_probabilities.shape[0] != len(batch):
                    raise ValueError('The stored classifier returned an invalid probability array.')
                probability_batches.append(raw_probabilities[:, class_column_indices])

            probabilities = np.vstack(probability_batches)
            if not np.isfinite(probabilities).all():
                raise ValueError('The stored classifier returned non-finite probabilities.')
            return probabilities

        means = np.vstack([np.mean(z[y_train == class_value], axis=0) for class_value in class_values])
        covariances = self._transformed_covariances(
            pooled_input_covariance=self._linear_input_covariance(
                classifier=classifier,
                fallback=np.asarray(projection.covariance_, dtype=np.float64),
            ) if use_fitted_oas_space else np.asarray(projection.covariance_, dtype=np.float64),
            scalings=scalings,
            z=z,
            y_true=y_train,
            class_values=class_values,
            dimensions=dimensions,
        )
        sample_probabilities = projected_probabilities(z)

        title_prefix = 'LDA' if self._decision_boundary == 'linear' else 'QDA'
        title = f'{title_prefix} projected into LDA-transformed space - {len(class_values)} classes, {dimensions} discriminant dimension(s)'

        if dimensions == 1:
            plot_values = self._prepare_1d_plot_data(z=z, means=means, covariances=covariances, probabilities=projected_probabilities)
        else:
            plot_values = self._prepare_2d_plot_data(
                z=z,
                means=means,
                covariances=covariances,
                probabilities=projected_probabilities,
                n_classes=len(class_values)
            )

        return DiscriminantPlotData(
            title=title,
            decision_boundary=self._decision_boundary,
            class_names=class_names,
            class_values=class_values.copy(),
            y_true=y_train.copy(),
            z=z,
            sample_probabilities=sample_probabilities,
            means=means,
            covariances=covariances,
            **plot_values
        )

    @staticmethod
    def _base_estimators(classifier) -> tuple:
        """Return fitted estimators from a classifier or calibration wrapper."""

        calibrated_classifiers = getattr(classifier, 'calibrated_classifiers_', None)
        if calibrated_classifiers is None:
            return (classifier,)

        estimators = tuple(
            calibrated.estimator
            for calibrated in calibrated_classifiers
            if getattr(calibrated, 'estimator', None) is not None
        )
        return estimators or (classifier,)

    @classmethod
    def _coefficient_projection_basis(cls, classifier, fallback_scalings: np.ndarray, n_components: int) -> np.ndarray:
        """ Construct a fitted OAS discriminant basis.

            Binary LDA has one coefficient row, which directly supplies its sole
            discriminant direction. For multiclass LDA, posterior probabilities
            depend on differences between class scores, so centering coefficient
            rows removes the irrelevant common score direction and yields at most
            ``n_classes - 1`` directions. Calibrated fold estimators are normalized
            per fold and stacked; the leading right-singular vectors form a stable
            aggregate space. Any missing rank is supplied by the visualization SVD
            LDA basis.
        """

        direction_blocks = []
        estimators = cls._base_estimators(classifier)
        aggregate_folds = len(estimators) > 1

        for estimator in estimators:
            coefficients = getattr(estimator, 'coef_', None)
            if coefficients is None:
                continue

            coefficients = np.asarray(coefficients, dtype=np.float64)
            if coefficients.ndim == 1:
                coefficients = coefficients[None, :]
            if (
                    coefficients.ndim != 2
                    or coefficients.shape[1] != fallback_scalings.shape[0]
                    or not np.isfinite(coefficients).all()
            ):
                continue

            if coefficients.shape[0] == 1:
                directions = coefficients
            else:
                directions = coefficients - np.mean(coefficients, axis=0, keepdims=True)

            block_norm = float(np.linalg.norm(directions))
            if block_norm <= 1e-12:
                continue
            if aggregate_folds:
                # Give each calibration fold equal influence rather than
                # allowing arbitrary coefficient magnitude to dominate.
                directions = directions / block_norm
            direction_blocks.append(directions)

        primary_basis = np.empty((fallback_scalings.shape[0], 0), dtype=np.float64)
        if direction_blocks:
            direction_matrix = np.vstack(direction_blocks)
            _, singular_values, right_vectors = np.linalg.svd(
                direction_matrix,
                full_matrices=False,
            )
            if singular_values.size and singular_values[0] > 0.0:
                tolerance = (
                    np.finfo(np.float64).eps
                    * max(direction_matrix.shape)
                    * singular_values[0]
                )
                rank = int(np.count_nonzero(singular_values > tolerance))
                primary_basis = right_vectors[:min(rank, n_components)].T

        return cls._augment_projection_basis(
            primary_basis=primary_basis,
            fallback_scalings=fallback_scalings,
            n_components=n_components,
        )

    @staticmethod
    def _augment_projection_basis(
            primary_basis: np.ndarray,
            fallback_scalings: np.ndarray,
            n_components: int,
    ) -> np.ndarray:
        """Orthonormalize a primary basis and fill missing axes from fallback."""

        feature_count = fallback_scalings.shape[0]
        vectors = []
        for candidates in (primary_basis, fallback_scalings):
            candidates = np.asarray(candidates, dtype=np.float64)
            if candidates.ndim != 2 or candidates.shape[0] != feature_count:
                continue

            for column in range(candidates.shape[1]):
                vector = candidates[:, column].copy()
                if not np.isfinite(vector).all():
                    continue
                for existing in vectors:
                    vector -= np.dot(existing, vector) * existing
                norm = float(np.linalg.norm(vector))
                if norm <= 1e-12:
                    continue
                vectors.append(vector / norm)
                if len(vectors) == n_components:
                    return np.column_stack(vectors)

        raise ValueError(
            f'Could not construct {n_components} independent discriminant projection directions.'
        )

    @classmethod
    def _linear_input_covariance(
            cls,
            classifier,
            fallback: np.ndarray,
    ) -> np.ndarray:
        """Average fitted OAS covariances, falling back to visualization LDA."""

        expected_shape = np.asarray(fallback).shape
        covariances = []
        for estimator in cls._base_estimators(classifier):
            covariance = getattr(estimator, 'covariance_', None)
            if covariance is None:
                continue
            covariance = np.asarray(covariance, dtype=np.float64)
            if covariance.shape == expected_shape and np.isfinite(covariance).all():
                covariances.append(covariance)

        if not covariances:
            return np.asarray(fallback, dtype=np.float64)
        return np.mean(np.stack(covariances), axis=0)

    def _transformed_covariances(
            self,
            pooled_input_covariance: np.ndarray,
            scalings: np.ndarray,
            z: np.ndarray,
            y_true: np.ndarray,
            class_values: np.ndarray,
            dimensions: int,
    ) -> np.ndarray:
        if self._decision_boundary == 'linear':
            pooled = scalings.T @ pooled_input_covariance @ scalings
            pooled = _positive_semidefinite_covariance(pooled, dimensions)
            return np.repeat(pooled[None, :, :], len(class_values), axis=0)

        covariances = []
        for class_value in class_values:
            class_samples = z[y_true == class_value]
            if len(class_samples) < 2:
                covariance = np.eye(dimensions, dtype=np.float64) * 1e-12
            else:
                covariance = np.cov(class_samples, rowvar=False)
            covariances.append(
                _positive_semidefinite_covariance(covariance, dimensions)
            )
        return np.stack(covariances)

    @staticmethod
    def _prepare_1d_plot_data(
            z: np.ndarray,
            means: np.ndarray,
            covariances: np.ndarray,
            probabilities: Callable[[np.ndarray], np.ndarray],
    ) -> dict:
        z1 = z[:, 0]
        x_min = float(np.min(z1))
        x_max = float(np.max(z1))
        x_span = max(1e-8, x_max - x_min)
        x_min -= 0.14 * x_span
        x_max += 0.14 * x_span

        gaussian_x = np.linspace(x_min, x_max, _GAUSSIAN_POINTS)
        gaussian_pdfs = np.vstack([
            _gaussian_pdf_1d(
                gaussian_x,
                mean=float(means[class_id, 0]),
                variance=float(covariances[class_id, 0, 0]),
            )
            for class_id in range(len(means))
        ])
        maximum_pdf = max(float(np.max(gaussian_pdfs)), 1e-8)
        y_min = -0.16 * maximum_pdf
        y_max = 1.18 * maximum_pdf

        background_width = max(700, _BACKGROUND_IMAGE_SIZE * 2)
        background_height = 220
        background_x = np.linspace(x_min, x_max, background_width)
        background_row = _probability_background_rgba(
            probabilities(background_x[:, None])
        )
        background_rgba = np.repeat(
            background_row[None, :, :],
            background_height,
            axis=0,
        )

        boundary_grid = np.linspace(x_min, x_max, 3000)
        boundary_probabilities = probabilities(boundary_grid[:, None])
        labels = np.argmax(boundary_probabilities, axis=1)
        transitions = np.flatnonzero(labels[:-1] != labels[1:])
        boundary_positions = []

        for transition in transitions:
            left = float(boundary_grid[transition])
            right = float(boundary_grid[transition + 1])
            left_class = int(labels[transition])
            right_class = int(labels[transition + 1])
            left_probability = boundary_probabilities[transition]
            left_difference = float(
                left_probability[left_class] - left_probability[right_class]
            )

            for _ in range(60):
                middle = 0.5 * (left + right)
                middle_probability = probabilities(np.asarray([[middle]]))[0]
                middle_difference = float(
                    middle_probability[left_class] - middle_probability[right_class]
                )
                if left_difference * middle_difference <= 0.0:
                    right = middle
                else:
                    left = middle
                    left_difference = middle_difference

            boundary_positions.append(0.5 * (left + right))

        return {
            'x_range': (x_min, x_max),
            'y_range': (y_min, y_max),
            'background_rgba': np.ascontiguousarray(background_rgba),
            'gaussian_x': gaussian_x,
            'gaussian_pdfs': gaussian_pdfs,
            'boundary_positions': np.asarray(boundary_positions, dtype=np.float64),
            'ellipses': (),
            'boundary_polylines': (),
        }

    @staticmethod
    def _prepare_2d_plot_data(
            z: np.ndarray,
            means: np.ndarray,
            covariances: np.ndarray,
            probabilities: Callable[[np.ndarray], np.ndarray],
            n_classes: int,
    ) -> dict:
        z2 = z[:, :2]
        x_min = float(np.min(z2[:, 0]))
        x_max = float(np.max(z2[:, 0]))
        y_min = float(np.min(z2[:, 1]))
        y_max = float(np.max(z2[:, 1]))
        x_span = max(1e-8, x_max - x_min)
        y_span = max(1e-8, y_max - y_min)
        x_min -= 0.14 * x_span
        x_max += 0.14 * x_span
        y_min -= 0.14 * y_span
        y_max += 0.14 * y_span

        background_x = np.linspace(x_min, x_max, _BACKGROUND_IMAGE_SIZE)
        # Image scanlines run top-to-bottom, hence descending Y values.
        background_y = np.linspace(y_max, y_min, _BACKGROUND_IMAGE_SIZE)
        background_xx, background_yy = np.meshgrid(background_x, background_y)
        background_query = np.column_stack((
            background_xx.ravel(),
            background_yy.ravel(),
        ))
        background_rgba = _probability_background_rgba(
            probabilities(background_query)
        ).reshape(_BACKGROUND_IMAGE_SIZE, _BACKGROUND_IMAGE_SIZE, 4)

        ellipses = tuple(
            _covariance_ellipse(means[class_id, :2], covariances[class_id, :2, :2])
            for class_id in range(n_classes)
        )

        boundary_x = np.linspace(x_min, x_max, _BOUNDARY_GRID_SIZE)
        boundary_y = np.linspace(y_min, y_max, _BOUNDARY_GRID_SIZE)
        boundary_xx, boundary_yy = np.meshgrid(boundary_x, boundary_y)
        boundary_query = np.column_stack((boundary_xx.ravel(), boundary_yy.ravel()))
        probability_grid = probabilities(boundary_query).reshape(
            _BOUNDARY_GRID_SIZE,
            _BOUNDARY_GRID_SIZE,
            n_classes,
        )

        boundary_polylines = []
        for first_class in range(n_classes):
            for second_class in range(first_class + 1, n_classes):
                scalar_field = (
                    probability_grid[:, :, first_class]
                    - probability_grid[:, :, second_class]
                )
                segments = _marching_squares_segments(
                    x_values=boundary_x,
                    y_values=boundary_y,
                    scalar_field=scalar_field,
                )
                if not segments:
                    continue

                # Pairwise equality contours are visible decision boundaries
                # only where those two classes are the two dominant posteriors.
                if n_classes > 2:
                    midpoints = np.asarray([
                        [
                            0.5 * (first[0] + second[0]),
                            0.5 * (first[1] + second[1]),
                        ]
                        for first, second in segments
                    ])
                    midpoint_probabilities = probabilities(midpoints)
                    top_two = np.argpartition(
                        midpoint_probabilities,
                        kth=n_classes - 2,
                        axis=1,
                    )[:, -2:]
                    keep = (
                        np.any(top_two == first_class, axis=1)
                        & np.any(top_two == second_class, axis=1)
                    )
                    segments = [
                        segment
                        for segment, is_visible in zip(segments, keep)
                        if bool(is_visible)
                    ]

                boundary_polylines.extend(_connect_segments(segments))

        return {
            'x_range': (x_min, x_max),
            'y_range': (y_min, y_max),
            'background_rgba': np.ascontiguousarray(background_rgba),
            'gaussian_x': np.empty(0, dtype=np.float64),
            'gaussian_pdfs': np.empty((n_classes, 0), dtype=np.float64),
            'boundary_positions': np.empty(0, dtype=np.float64),
            'ellipses': ellipses,
            'boundary_polylines': tuple(boundary_polylines),
        }
