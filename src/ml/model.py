"""
Train and evaluate a model to predict the income category on the census data.

Created on Thu Apr 18 10:39:17 2024

@author: juergen
"""

from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

from utils import get_logger

logger = get_logger(__name__)


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train, model_parameters={}):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    ml_model
        Trained machine learning model.
    """
    ml_model = RandomForestClassifier(**model_parameters)
    ml_model.fit(X_train, y_train)
    return ml_model


def compute_model_metrics(y_true, y_pred):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y_true : np.array
        Known labels, binarized.
    y_pred : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    try:
        precision = precision_score(y_true, y_pred, zero_division=1)
        recall = recall_score(y_true, y_pred, zero_division=1)
        fbeta = fbeta_score(y_true, y_pred, beta=1, zero_division=1)
    except ValueError as ex:
        logger.info("an exception was encountered, %s", ex)
        if hasattr(y_true, "shape"):
            logger.info("shape of y_true: %s", y_true.shape)
        else:
            logger.info("len of y_true: %s", len(y_true))
        if hasattr(y_pred, "shape"):
            logger.info("shape of y_pred: %s", y_pred.shape)
        else:
            logger.info("len of y_pred: %s", len(y_pred))
        raise ex
    return precision, recall, fbeta


def inference(ml_model, X):
    """
    Run model inferences and return the predictions.

    Inputs
    ------
    ml_model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return ml_model.predict(X)
