import numpy as np


def get_ensemble_predictions(x: np.ndarray, models: list) -> (np.ndarray, np.ndarray):
    sum_predict_proba = np.zeros(shape=(x.shape[0], 3), dtype=np.float64)

    for model in models:
        _, predict_proba = model.predict(x=x)
        sum_predict_proba += predict_proba
    y_prob = sum_predict_proba / len(models)
    y_pred = np.argmax(y_prob, axis=1)
    return y_pred, y_prob
