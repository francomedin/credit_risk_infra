# Python standard Library
import time

# Third party libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

# Local Libraries
from model.scripts.preprocessing import get_data_preprocessed


def get_performance(y_test, predictions):
    """
    Calculates several metrics to see model perfomance
    Parameters
    ----------
    y_test : pd.DataFrame
        Target values to test models.
    predictions : numpy.array
        Prediction made by the model.
    Returns
    -------
    accuracy = float
    precision = float
    recall = float
    f1_score = float
        Print the metrics, report and the confusion matrix.
    """
    accuracy = metrics.accuracy_score(y_test, predictions)
    precision = metrics.precision_score(y_test, predictions)
    recall = metrics.recall_score(y_test, predictions)
    f1_score = metrics.f1_score(y_test, predictions)
    report = metrics.classification_report(y_test, predictions)

    cm = metrics.confusion_matrix(y_test, predictions)
    cm_as_dataframe = pd.DataFrame(data=cm)

    print("Model Performance metrics:")
    print("-" * 30)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_score)
    print("\nModel Classification report:")
    print("-" * 30)
    print(report)
    print("\nPrediction Confusion Matrix:")
    print("-" * 30)
    print(cm_as_dataframe)

    return accuracy, precision, recall, f1_score


def get_roc(model, y_test, features):
    """
    Calculates the roc auc metric
    Parameters
    ----------
    model : object
        AI model.
    y_test : pd.DataFrame
        Target values to test models
    features:
        Features to predict target value
    Returns
    -------
    roc_auc = float
        Metric to evaluate our model.
    """
    try:
        if type(model).__name__ == 'Sequential':
            roc_auc = metrics.roc_auc_score(y_test, model.predict_proba(features))
        else:
            roc_auc = metrics.roc_auc_score(y_test, model.predict_proba(features)[:, 1])
    except:
        roc_auc = metrics.roc_auc_score(y_test, model.predict(features))
    return roc_auc


def plot_roc(model, y_test, features):
    """
    Plot Roc curve

    Parameters
    ----------
    model : object
        AI model.
    y_test : pd.DataFrame
        Target values to test models
    features:
        Features to predict target value
    Returns
    -------
    roc_auc = float
        Metric to evaluate our model.
    """
    fpr = None
    tpr = None
    if type(model).__name__ == 'Sequential':
        fpr, tpr, _ = metrics.roc_curve(y_test, model.predict(features))
        roc_auc = metrics.roc_auc_score(y_test, model.predict(features))
    else:
        fpr, tpr, _ = metrics.roc_curve(y_test, model.predict_proba(features)[:, 1])
        roc_auc = metrics.roc_auc_score(y_test, model.predict_proba(features)[:, 1])
    
    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc})", linewidth=2.5)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc


def get_model_metrics(path, model):
    """
    End to End preprocessing and model evaluation.

    Parameters
    ----------
    path : str
        Path to the dataset.
    model : object
        AI model.

    """
    X_train, x_test, y_train, y_test = get_data_preprocessed(path)
    print("Training starts: ")
    model_name = type(model).__name__
    print(f"Model name: {model_name}")
    start_training = time.time()
    if model_name == 'Sequential':
        model.fit(X_train, y_train, epochs = 3)
    else:
        model.fit(X_train, y_train)
    end_training = time.time()
    training_time = end_training - start_training
    print(f"Training time {round(training_time,5)} seconds")
    print("Prediction starts")
    start_predict = time.time()
    if model_name == 'Sequential':
        y_predict_proba = model.predict(x_test)
        y_predicted = np.where(y_predict_proba > 0.5, 1, 0)
    else:
        y_predicted = model.predict(x_test)
    end_predict = time.time()
    predicting_time = end_predict - start_predict
    print(f"Predicting time {round(predicting_time,5)} seconds")
    print("Metrics: ")
    get_performance(y_test, y_predicted)
    print("-" * 100)
    roc_auc = plot_roc(model, y_test, x_test)
    print("-" * 100)

    print(f"AUC ROC Result: {roc_auc}")
