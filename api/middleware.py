# Python standard Library
import time
import uuid
import json

# Third party libraries
import redis

# Local Libraries
import settings

db = redis.Redis(
host=settings.REDIS_IP, 
port=settings.REDIS_PORT, 
db=settings.REDIS_DB_ID
)



def model_predict(application):
    """
    Receives an application and queues the job into Redis.
    Will loop until getting the answer from our ML service.

    Parameters
    ----------
    application : dict
        dict with features values.

    Returns
    -------
    prediction, prediction_proba : tuple(float, float)
        Model predicted target as a float and the corresponding confidence
        score as a number.
    """
   
    job_id = str(uuid.uuid4())
    job_data =  {
    "id": job_id,
    "application": application,
    }
    job_data = json.dumps(job_data)
    
    db.rpush(settings.REDIS_QUEUE, job_data)
    # Loop until we received the response from our ML model
    while True:
        if db.exists(job_id):
            model_prediction = db.get(job_id)
            db.delete(job_id)
            break
        time.sleep(settings.API_SLEEP)
    dict_pred = json.loads(model_prediction)

    return dict_pred["prediction"], dict_pred["prediction_proba"]


    
