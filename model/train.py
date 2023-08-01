# Python standard Library
import pickle
import argparse
import os

# Third party libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

from keras.models import Sequential
from keras.layers import Dense

# Local Libraries
import grid_config
from model.evaluation import get_performance, get_roc
from scripts.preprocessing import get_data_preprocessed


def parse_args():
    """
    Receives arguments in the command line.

    Parameters
    ----------

    Returns
    -------
    args object
        With the variables taken from de command line.
    """
    parser = argparse.ArgumentParser(description="Train your model.")

    parser.add_argument(
        "from_folder",
        type=str,
        nargs='?',
        default="data/complete_data.csv",

        help=(
            "Full path to the CSV file with data E.g. " "src/data/complete_data.csv."
        ),
    )

    parser.add_argument(
        "grid_search",
        type=str,
        nargs='?',
        default="No",
        help=("Yes or No to load gridsearch from grid_config.py "),
    )

    parser.add_argument(
        "model_name",
        type=str,
        nargs='?',
        default="LightGBM",
        help=(
            "Model choosen to train i.e." "lightgbm, randomforest, catboost, enssemble"
        ),
    )

    parser.add_argument(
        "cross_validation",
        type=int,
        nargs='?',
        default=3,
        help=("Number of Cross Validations "),
    )

    parser.add_argument(
        "n_iter",
        type=int,
        nargs='?',
        default=3,
        help=("Number of iteration"),
    )

    args = parser.parse_args()

    return args


def save_model(model_name, model, auc_roc):
    """
    Saves the model into a pickle file if it superates a treshold of 0.62 in auc roc.

    Parameters
    ----------
    model_name : str
        Name of the model.
    model : object
        Machine learning model.
    auc_roc : float

    """
    if auc_roc > 0.62:
        try:
            auc_roc = round(auc_roc, 5)
            path = os.path.join("model/Experiments/", f"{model_name}_{auc_roc}")
            if not os.path.exists(path):
                # Creates a folder to save the model
                print("Model with same scoring does not exist")
                os.umask(0)
                os.makedirs(path, mode=0o777)
                with open(f"{path}/{model_name}.{auc_roc}.pickle", "wb") as file:
                    pickle.dump(model, file)
                try:
                    model_results = pd.DataFrame(model.cv_results_).sort_values(
                        by="rank_test_score", ascending=True
                    )
                    model_results.to_csv(f"{path}/{model_name}.{auc_roc}.csv")
                except:
                    print("Model does not use a grid search object")
                print(f"Model {model_name} saved")
            else:
                print("The same score model already exists")
        except Exception as ex:
            print(f"Error saving the model, error {type(ex).__name__}, {ex.args}")
    else:
        print(f"Model score {auc_roc} is to low to be saved.")


def select_model(model_name):
    """
    Receives a Machine Learning model name and select it from others.

    Parameters
    ----------
    model_name : str
        Name for the ML model to be selected.

    Returns
    -------
    model = object
        ML model.
    """
    print(f"Selecting model: {model_name}")
    model_name = model_name.lower()
    try:
        if model_name == "logistic":
            model = LogisticRegression(solver="lbfgs", max_iter=1000)
        elif model_name == "lightgbm":
            model = lgb.LGBMClassifier()
        elif model_name == "randomforest":
            model = RandomForestClassifier()
        elif model_name == "adaboost":
            model = AdaBoostClassifier(n_estimators=100)
        elif model_name == "xgboost":
            model = XGBClassifier(
                objective="binary:logistic",
                booster="gbtree",
                eval_metric="auc",
                tree_method="hist",
                grow_policy="lossguide",
                use_label_encoder=False,
            )
        elif model_name == "catboost":
            model = CatBoostClassifier()
        elif model_name == "neuronalnetwork":
            model = Sequential()
            model.add(Dense(743, input_shape=(743,), activation='relu'))
            model.add(Dense(743,activation="relu"))
            model.add(Dense(1, activation='sigmoid'))
            # Compile model
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])
            return model





           
        elif model_name == "enssemble":
            clf1 = RandomForestClassifier(random_state=1)
            clf2 = lgb.LGBMClassifier(random_state=1)
            clf3 = XGBClassifier()
            clf4 = GaussianNB()
            clf5 = CatBoostClassifier(
                depth=6, iterations=100, learning_rate=0.05, random_state=1
            )
            eclf = VotingClassifier(
                estimators=[
                    ("rf", clf1),
                    ("lg", clf2),
                    ("xgb", clf3),
                    ("gnb", clf4),
                    ("cat", clf5),
                ],
                voting="soft",
                weights=[1, 2, 2, 1, 1],
            )
            model = eclf

    except:
        print("No model was found")
    return model


def get_grid(model_name):
    """
    Receives a model name and returns a grid of hyperparameters..

    Parameters
    ----------
    model_name : str
        Name of de model.

    Returns
    -------
    grid : Dict

    """
    print("Getting grid of values")
    model_name = model_name.lower()
    try:
        if model_name == "lightgbm":
            grid = grid_config.LIGHTGBM
        elif model_name == "randomforest":
            grid = grid_config.RANDOM_FOREST
        elif model_name == "enssemble":
            grid = grid_config.ENSSEMBLE
        elif model_name == "xgboost":
            grid = grid_config.XGBOOST
        elif model_name == "catboost":
            grid = grid_config.CATBOOST
    except:
        print("No grid was found")

    return grid


def train(path_to_csv, grid_search, model_name, cross_validation, n_iter):
    """
    Train one model choosen by the user, using or not grid search
    with the data from the path. Also shows the model´s metrics.
    Finally if the model´s metrics are good enough, saves the model into a pickle object.


    Parameters
    ----------
    path_to_csv : str
    grid_search : str
    model_name : str
    cross_validation : int
    n_iter : int

    """
    X_train, x_test, y_train, y_test = get_data_preprocessed(path_to_csv)
    print(X_train.shape)
    

    
    model = select_model(model_name)

    if grid_search == "Si":
        model_grid_search = RandomizedSearchCV(
            model,
            param_distributions=get_grid(model_name),
            random_state=10,
            scoring="roc_auc",
            cv=cross_validation,
            n_iter=n_iter,
            verbose=3,
        )
        model = model_grid_search.fit(
            X_train,
            y_train,
        )
        y_predict = model.predict(x_test)
    else:
        if model_name == 'neuronalnetwork':
            model.fit(X_train, y_train, epochs = 5, batch_size = 500)
            print('neuronal_network')
            y_predict_proba = model.predict(x_test)
            y_predict = np.where(y_predict_proba > 0.5, 1, 0)
        else:
            model.fit(X_train, y_train,)
            y_predict = model.predict(x_test)
    

    roc_auc_score = get_roc(model, y_test, x_test)
    print(f"Result {roc_auc_score}")
    get_performance(y_test, y_predict)

    save_model(model_name, model, roc_auc_score)


if __name__ == "__main__":
    # Now launch process
    print("Launching ML service...")
    args = parse_args()
    train(args.from_folder, args.grid_search, args.model_name, args.cross_validation, args.n_iter)
