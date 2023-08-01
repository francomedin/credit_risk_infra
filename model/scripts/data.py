# Python standard Library
import os
import zipfile

# Third party libraries
import pandas as pd
import boto3
from cloudpathlib import CloudPath
from dotenv import load_dotenv

load_dotenv()

def get_data():
    """
    Obtain the data from an AWS bucket then the data is unziped 
    and applies changes into some feartures.
    Finally store all the data into data folder.
    
    """
    # fetch credentials from env variables
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    
    # setup a AWS S3 client/resource
    s3 = boto3.resource(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    
    bucket = s3.Bucket('anyoneai-ay22-01')
    
    #Create data folder
    if not os.path.exists('/data'):
        os.mkdir('data')
        
    #print all object names found in the bucket
    print('Existing buckets:')
    for file in bucket.objects.filter(Prefix="credit-data-2010"):
        print(file, flush = True)
    
    
    #download dataset
    dataset = CloudPath("s3://anyoneai-datasets/credit-data-2010/")
    dataset.download_to("data")
    
    # Extract files
    zip = zipfile.ZipFile('data/PAKDD-2010 training data.zip')
    zip.extractall('data/') 
    zip.close()
    variable_description = pd.read_excel('data/PAKDD2010_VariablesList.XLS')
    labels = variable_description['Var_Title'].tolist()
    labels[43] = "MATE_EDUCATION_LEVEL"
    labels[-1] = "TARGET"
    modeling_ds = pd.read_csv('data/PAKDD2010_Modeling_Data.txt', encoding ='ISO-8859-1',sep='\t', header = None )
    modeling_ds.columns = labels
    modeling_ds.shape
    modeling_ds.drop('ID_CLIENT', axis = 1 , inplace=True)
    modeling_ds.to_csv('data/complete_data.csv', index = False)


if __name__ == "__main__":
    get_data()


