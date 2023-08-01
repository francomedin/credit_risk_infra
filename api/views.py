# Python standard Library
import json

# Third party libraries
from flask import (
    Blueprint,
    render_template,
    request,
    jsonify
)

# Local Libraries
from middleware import model_predict
import settings


router = Blueprint("app_router", __name__, template_folder="templates")


@router.route("/", methods=["GET"])
def index():
    """
    Render the index.html where the application form is.

    Returns
    -------
    Render the index.html
    """
    return render_template("index.html")


@router.route("/application", methods=["POST"])
def application():
    """
    Receives the application form and convert its values into a dict. Then send the dict
    though model_predict() to redis and wait for the response. Finally renderize the score in
    application_success.html.

    Returns
    -------
    prediction, prediction_proba : tuple(float, float)
        Render the application_success.html with the model score.
    """
    
    req= request.form.to_dict(flat=False)
    dict_well = {}
    for element in req:
        dict_well[element] = req[element][0]
    prediction, prediction_proba = model_predict(dict_well)
    print(dict_well,flush=True)
    context = {
        "prediction": prediction,
        "prediction_proba": round(prediction_proba *100 ,2)    
    }
    if round(prediction_proba *100 ,2) < settings.TRESHOLD:
        return render_template("application_success.html", context=context)
    else:
        return render_template("application_denied.html", context=context)




@router.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint used to get predictions without need to access the UI.

    Parameters
    ----------
    req : dict
        Input dict we want to get predictions from.

    Returns
    -------
    flask.Response
        JSON response from our API having the following format:
            {
                "success": bool,
                "prediction": int,
                "prediction_proba": float,
            }

        - "success" will be True if the input file is valid and we get a
          prediction from our ML model.
        - "prediction" model predicted target as int.
        - "score" model confidence score for the predicted target as float.
    """
    try:
        rpse = {"success": False, "prediction": None, "score": None}
        # Request object is a json.
        req= request.get_json()
        print(req,flush=True)
        prediction, prediction_proba = model_predict(req)
        rpse = {"success": True, "prediction": prediction, "prediction_proba": prediction_proba}
        
        return jsonify(rpse)
    except:
        return jsonify(rpse), 400

  
 

