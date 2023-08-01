# CreditFocus
## _Give loans based on default probability_



CreditFocus is Artificial Inteligence archichecture, ready to deploy for credit risk analyisis.

## Features

- Download data from AWS Bucket.
- Train several ML models.
- Scalability through an infrastructure based on Containers and Redis.
- Hyperparameter tuning with grid search (Optional).
- GridSearch Customization.
- Two endpoints: A-Integrated with a UI and B-Ready to be consumed.



## Tech

Dillinger uses a number of open source projects to work properly:

-  Python
- Pandas, Numpy, Seaborn.
- Scikit Learn
- TensorFlow
- DockerHub
- JupyterNotebooks.
- Amazon Web Services
- Redis

## Docker
CreditFocus is very easy to set up and deploy in a Docker container.

By default, the Docker will expose port 8080, so change this within the
Dockerfile if necessary. When ready, simply use the Dockerfile to
build the image.

```sh
cd credit_risk_data
docker build -t credit_focus -f Dockerfile .
```

This will create the credit_focus image and pull in the necessary dependencies.

Once done, run the Docker image and map the port to whatever you wish on
your host. In this example, we simply map port 8000 of the host to
port 8080 of the Docker (or whatever port was exposed in the Dockerfile):

```sh
docker run --rm -it -p 8888:8888 -v $(pwd):/home/app/src credit_focus bash
```
.

Verify the deployment by navigating to your server address in
your preferred browser.

```sh
127.0.0.1:8000
```

## Installation

credit_focus requires [Python](https://www.python.org/downloads/release/python-3811/) 3.8 to run.



### For Train models

1.  _Run docker_
```sh
docker run --rm -it -p 8888:8888 -v $(pwd):/home/app/src credit_focus 
```
2. _Download data from AWS_

After execute the scrip you have to export the keys to access AWS.

```sh
python model/scripts/data.py
```
This will install create the folder data and download all the files in it.

3. _Train Models_
Here you can set up 4 parameters
a-  Full path to the data, default = 'data/complete_data.csv'
b- Grid Search default = 'No', options = 'Yes', 'No'
c- Model Name default = 'LightGBM', options = 'logistic', 'lightgbm', 'randomforest', 'adaboost', 'xgboost', 'catboost', 'neuronalnetwork'
d- Cross Validation = 3 options= 3, 6 , 25 ...
e- Number of iteration = 3 options = 3, 6 , 25 ...


```sh
python model/train.py
```

After train, if the model is above a 0.62 auroc will be saved in the "Experiment" folder. With the followin name f"{model_name}_{auc_roc}.pickle" as a picke file.


Finally, grid_config.py has all the grids to use in RandomizedSearch. You can change it and try several combinations.
For production environments...

### Use credit_focus
After train and choose one model is necesary to write the path in the models/ml_service.py shoul be like the followin line
_model = pickle.load(open('Experiments/model_name/model_name.pickle', 'rb'))_

After this the project is ready to run credit_focus through docker.

```sh
docker-compose up --build
```

This will install all the requirements needed by each part
1. Api
2. Redis
2. Ml Service

After installation you can open your localhost/ an try an application example.
