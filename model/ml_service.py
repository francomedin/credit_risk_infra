# Python standard Library
import time
import pickle
import redis
import json
import os

# Third party libraries
import pandas as pd

# Local Libraries
from scripts.constants import ALL_FEATURES
from scripts.preprocessing import preprocess_application
import settings

db = redis.Redis(
host=settings.REDIS_IP, 
port=settings.REDIS_PORT, 
db=settings.REDIS_DB_ID
)

# Load your ML model and assign to variable `model`
print('Loading Model ...')


model = pickle.load(open('Experiments/LightGBM_0.63984/LightGBM.0.63984.pickle', 'rb'))


def predict(application):
    """
    Receives the application form and run our ML model to get predictions.
    

    Parameters
    ----------
    application : pandas.DataFrame
        Dataframe with the application data.

    Returns
    -------
    prediction = Model target predicted.
    prediction_proba : Model probability of target = 1 predicted.
    """
    app_preprocessed =  preprocess_application(application)
    
   

    prediction = model.predict(app_preprocessed)[0]
    prediction_proba = model.predict_proba(app_preprocessed)[:,1][0]

    return prediction, prediction_proba
  

def save_application(application):
    """
    Receives the application dataframe and save it into the existing csv.
    

    Parameters
    ----------
    application : pandas.DataFrame
        Dataframe with the application data.

    """
    path = 'applications/applications.csv'
    try:
        if os.path.exists(path):
            
            applications = pd.read_csv(path)
            applications = pd.concat([applications, application], ignore_index=True)
            print(applications.shape,flush = True)

            applications.to_csv(path, index = False)
        else:
            application.to_csv(path, index = False)
    except:
        print('Error saving new application', flush=True)

def classify_process():
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.
    """
    while True:
        _, work = db.brpop(settings.REDIS_QUEUE)
        work_dict = json.loads(work)
        application = work_dict['application']
        value_list = list(application.values())
        new_application = pd.DataFrame([value_list], columns = (['NAME','DNI'] + ALL_FEATURES))
        save_application(new_application)
        new_application.drop('NAME', axis = 1, inplace = True)
        new_application.drop('DNI', axis = 1, inplace = True)
        prediction, prediction_proba = predict(new_application)
        dict_predicted = {
        "prediction": int(prediction),
        "prediction_proba": float(prediction_proba)
        }
        db.set(work_dict['id'], json.dumps(dict_predicted))
        # Don't forget to sleep for a bit at the end
        time.sleep(settings.SERVER_SLEEP)

if __name__ == "__main__":
    # Now launch process
    print("Launching ML service...")
    classify_process()
