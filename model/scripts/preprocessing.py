# Python standard Library
import sys
import os
import pickle

# Third party libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Local Libraries
try:
    from scripts import constants
except:
    try:
        from model.scripts import constants
    except:
        print('Error importing constants')

def drop_columns(dataset):
    """
    Deletes columns already defined in constants.py from de dataset.

    Parameters
    ----------
    dataset : pandas.DataFrame
        pandas dataframe.
    Returns
    -------
    dataframe
        Dataframe with columns dropped.
    """
    raw_data = dataset.copy()
    raw_data.drop(constants.UNIQUE_VALUES_FEATURES, axis=1, inplace=True)
    raw_data.drop(constants.LOCALIZATION_FEATURES, axis=1, inplace=True)
    raw_data.drop(constants.OTHERS_COLUMNS_DROPPED, axis=1, inplace=True)
    return raw_data


def apply_changes_numeric(dataset):
    """
    Applies changes into numeric columns already defined in constanst.py

    Parameters
    ----------
    dataset : pandas.DataFrame


    Returns
    -------
    dataset : pandas.DataFrame
        With several changes in numeric columns
    """
    raw_data_num = dataset.copy()
    # Max value for Quant_dependants set at 10
    raw_data_num["QUANT_DEPENDANTS"] = pd.to_numeric(
        raw_data_num["QUANT_DEPENDANTS"], downcast="float", errors="coerce"
    ).astype("int64")

    raw_data_num["QUANT_DEPENDANTS"] = raw_data_num["QUANT_DEPENDANTS"].apply(
        lambda x: x if x <= 15 else 15
    )
    try:
        # Casting to replace str valus to float64
        raw_data_num["MONTHS_IN_RESIDENCE"] = raw_data_num["MONTHS_IN_RESIDENCE"].fillna(raw_data_num["MONTHS_IN_RESIDENCE"].mean())
        raw_data_num["MONTHS_IN_RESIDENCE"] = pd.to_numeric(
            raw_data_num["MONTHS_IN_RESIDENCE"], downcast="float", errors="coerce"
        ).astype("int")

        raw_data_num["QUANT_BANKING_ACCOUNTS"] = pd.to_numeric(
            raw_data_num["QUANT_BANKING_ACCOUNTS"], downcast="float", errors="coerce"
        ).astype("int")

        raw_data_num["MONTHS_IN_THE_JOB"] = pd.to_numeric(
            raw_data_num["MONTHS_IN_THE_JOB"], downcast="float", errors="coerce"
        ).astype("int")

        raw_data_num["AGE"] = pd.to_numeric(
            raw_data_num["AGE"], downcast="float", errors="coerce"
        ).astype("int")

        raw_data_num["PERSONAL_MONTHLY_INCOME"] = pd.to_numeric(
            raw_data_num["PERSONAL_MONTHLY_INCOME"], downcast="float", errors="coerce"
        ).astype("float64")

        raw_data_num["OTHER_INCOMES"] = pd.to_numeric(
            raw_data_num["OTHER_INCOMES"], downcast="float", errors="coerce"
        ).astype("float64")

        raw_data_num["PERSONAL_ASSETS_VALUE"] = pd.to_numeric(
            raw_data_num["PERSONAL_ASSETS_VALUE"], downcast="float", errors="coerce"
        ).astype("float64")
    except:
        print("Error tryng to convert numeric columns to float")
    # Putting limits to the age of people who get a credit
    raw_data_num["AGE"] = raw_data_num["AGE"].apply(lambda x: x if x < 75 else 75)
    raw_data_num["AGE"] = raw_data_num["AGE"].apply(lambda x: x if x > 16 else 17)
    return raw_data_num


def apply_changes_categorical(dataset):
    """
    Applies changes into categorical columns already defined in constanst.py

    Parameters
    ----------
    dataset : pandas.DataFrame


    Returns
    -------
    dataset : pandas.DataFrame
        With several changes in categorical columns
    """
    raw_data_cat = dataset.copy()
    try:
        # Replace Y and N for 1 and 0
        replace_y_n = lambda x: 1 if x == "Y" else 0
        for column in constants.TWO_VALUE_LIST:
            if raw_data_cat[column].unique()[0] in ["Y", "N"]:
                
                raw_data_cat[column] = raw_data_cat[column].apply(replace_y_n)
    except:
        print("No Y or N values in TWO_VALUE_LIST features")

    # New category in submission type and sex
    try:
        raw_data_cat["APPLICATION_SUBMISSION_TYPE"] = raw_data_cat[
            "APPLICATION_SUBMISSION_TYPE"
        ].str.replace("0", "Posted")
        raw_data_cat["SEX"] = raw_data_cat["SEX"].str.replace(" ", "N/A")
    except:
        print("No 0 or " "  values in APPLICATION_SUBMISSION_TYPE or SEX ")

    # Replacing empty values
    try:
        raw_data_cat["STATE_OF_BIRTH"] = raw_data_cat["STATE_OF_BIRTH"].replace(
            r" ", np.NaN
        )
        raw_data_cat["STATE_OF_BIRTH"] = raw_data_cat["STATE_OF_BIRTH"].replace(
            r"XX", np.NaN
        )
    except:
        print("No empty or XX values in STATE_OF_BIRTH")
    return raw_data_cat


def zip_code(dataset, less_than=5):
    """
    Limit the number of minumun values into de zip_code.
    Finally, saves the data frame as csv with name: 'preprocessed_data.csv' "

    Parameters
    ----------
    dataset : pandas.DataFrame
    less_than : int


    Returns
    -------
    dataset : pandas.DataFrame
        With number of repeated zips greater than parameter less_than.
    """
    data_zipped = dataset.copy()
    
    # Replace not number str for nans
    replace_str = lambda x: 1 if type(x) == str and x.isdigit() == False else x
    data_zipped["RESIDENCIAL_ZIP_3"] = data_zipped["RESIDENCIAL_ZIP_3"].apply(
        replace_str
    )
    data_zipped["RESIDENCIAL_ZIP_3"].fillna(1, inplace=True)
    # Converting to int zip code
    data_zipped["RESIDENCIAL_ZIP_3"] = pd.to_numeric(
        data_zipped["RESIDENCIAL_ZIP_3"], downcast="float", errors="coerce"
    ).astype("int64")
    # Lambda Function to replace zip with just one value.
    replace_one = (
        lambda x: x
        if len(data_zipped[data_zipped["RESIDENCIAL_ZIP_3"] == x]) > less_than
        else np.nan
    )
    data_zipped["RESIDENCIAL_ZIP_3"] = data_zipped["RESIDENCIAL_ZIP_3"].apply(
        replace_one
    )
    # Create a new category with < less_than zips,
    data_zipped["RESIDENCIAL_ZIP_3"].fillna(1, inplace=True)

    return data_zipped


def split_data(dataset, train_size=0.8, test_size=0.2):
    """
    Receives a dataset and then Split the data into train and test. Taking the division
    percentage as a parameter.

    Parameters
    ----------
    dataset : pandas.DataFrame
    train_size : float
    test_size : float

    Returns
    -------
    X_train
    x_test
    y_train
    y_test
        All pandas.DataFrame
    """
    print("Spliting the data")
    X_train, x_test = train_test_split(
        dataset,
        random_state=5,
        train_size=train_size,
        test_size=test_size,
        stratify=dataset["RESIDENCIAL_ZIP_3"],
    )
    y_train = X_train["TARGET"]
    y_test = x_test["TARGET"]
    X_train.drop("TARGET", axis=1, inplace=True)
    x_test.drop("TARGET", axis=1, inplace=True)

    return X_train, x_test, y_train, y_test


def preprocessor(X_train, x_test):
    """
    Creates a pipeline to inpute nans and then encode variables. It separates categorical
    and numerica features. Then preprocess data and saves the preprocessor into a pickle object.

    Parameters
    ----------
    X_train : pandas.DataFrame
    x_test : pandas.DataFrame


    Returns
    -------
    X_train:  pandas.DataFrame
    x_test: pandas.DataFrame

    Boths dataset preprocessed.
    """
   
    numeric_features = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    
    categorical_features = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(sparse=False, handle_unknown="ignore")),
        ]
    )
   

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric_features", numeric_features, constants.NUMERICAL_FEATURES),
            (
                "categorical_features",
                categorical_features,
                constants.CATEGORICAL_FEATURES,
            ),
        ]
    )
    try:
        X_train = preprocessor.fit_transform(X_train)
        x_test = preprocessor.transform(x_test)
    except:
        print("Error trying to preprocess")

    # Saving preprocessor as pickle.
    path = os.path.join("model/scripts/tools", "preprocessor.pickle")
    if not os.path.exists(path):
        try:
            with open("model/scripts/tools/preprocessor.pickle", "wb") as file:
                pickle.dump(preprocessor, file)
        except: 
            pass

    return X_train, x_test


def get_dataset(path):
    """
    Checks if the path to dataset exists.
    If exists return dataset, else shows the error.

    Parameters
    ----------
    path : str

    Returns
    -------
    dataset: pd.DataFrame
    """
    print("Getting data")
    if os.path.exists(path):
        dataset = pd.read_csv(path)
    else:
        dataset = None
        print(f"Error getting the data, not file in path {path}")

    return dataset


def get_data_preprocessed(path):
    """
    Applies all preprocessing functions to obtain the data ready for train models.

    Parameters
    ----------
    path : str

    Returns
    -------
    X_train
    x_test
    y_train
    y_test
        All returns are pandas.DataFrames ready to be used in model training.
    """
    dataset = get_dataset(path)
    ds_drop = drop_columns(dataset)
    ds_numeric = apply_changes_numeric(ds_drop)
    ds_categorical = apply_changes_categorical(ds_numeric)
    ds_clean = zip_code(ds_categorical)

    X_train, x_test, y_train, y_test = split_data(ds_clean)
    X_train, x_test = preprocessor(X_train, x_test)
    return X_train, x_test, y_train, y_test


def preprocess_application(application):
    """
    Applies all preprocessing functions to a single application received through the API.

    Parameters
    ----------
    application : pandas.DataFrame

    Returns
    -------
    app_preprocessed : pandas.DataFrame
        pandas.DataFrame with the applicacion preprocessed.

    """
    try:
        preprocessor = pickle.load(open("scripts/tools/preprocessor.pickle", "rb"))
    except:
        sys.exit("Fail to load preprocessor, system will execute")

    app_cleaned = drop_columns(application)
    app_numerical = apply_changes_numeric(app_cleaned)
    app_categorical = apply_changes_categorical(app_numerical)
    app_zip = zip_code(app_categorical)
    app_preprocessed = preprocessor.transform(app_zip)
    return app_preprocessed


if __name__ == "__main__":
    # Now launch process
    print("Launching Pre-processing...")
    get_data_preprocessed()
