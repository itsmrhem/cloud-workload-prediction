from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import boto3
from datetime import datetime, timedelta, timezone
import json
from models.lstm_predictor_percent import CPUPercentagePredictor as LSTMPredictor
import os
import tensorflow as tf
import joblib
import numpy as np

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

aws_credentials = {"access_key": None, "secret_key": None}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/set-credentials")
async def set_credentials(access_key: str = Form(...), secret_key: str = Form(...)):
    aws_credentials["access_key"] = access_key
    aws_credentials["secret_key"] = secret_key
    return {"message": "Credentials set successfully"}

@app.get("/verify-credentials")
async def verify_credentials():
    if not aws_credentials["access_key"] or not aws_credentials["secret_key"]:
        return {"success": False, "error": "AWS credentials not set"}
    
    try:
        ec2 = boto3.client(
            'ec2',
            aws_access_key_id=aws_credentials["access_key"],
            aws_secret_access_key=aws_credentials["secret_key"]
        )
        response = ec2.describe_regions()
        print("\n=== AWS Credential Verification Response ===")
        print(json.dumps(response, indent=2, default=str))
        print("========================================\n")
        return {"success": True, "message": "Credentials are valid"}
    except Exception as e:
        print("\n=== AWS Credential Verification ERROR ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("=====================================\n")
        return {"success": False, "error": str(e)}

@app.get("/fetch-logs")
async def fetch_logs():
    if not aws_credentials["access_key"] or not aws_credentials["secret_key"]:
        return {"success": False, "error": "AWS credentials not set. Please set credentials first."}
    
    try:
        cloudwatch = boto3.client(
            'cloudwatch',
            aws_access_key_id=aws_credentials["access_key"],
            aws_secret_access_key=aws_credentials["secret_key"]
        )

        # Get data from 15 months ago (maximum CloudWatch retention) until now
        start_time = datetime.now(timezone.utc) - timedelta(days=455)  # ~15 months
        response = cloudwatch.get_metric_data(
            MetricDataQueries=[
                {
                    'Id': 'cpu_util',
                    'MetricStat': {
                        'Metric': {
                            'Namespace': 'AWS/EC2',
                            'MetricName': 'CPUUtilization',
                        },
                        'Period': 300,
                        'Stat': 'Average'
                    },
                    'ReturnData': True
                }
            ],
            StartTime=start_time,
            EndTime=datetime.now(timezone.utc)
        )
        
        print("\n=== CloudWatch Response ===")
        print(json.dumps(response, indent=2, default=str))
        print("==========================\n")

        return {"success": True, "data": response['MetricDataResults']}
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_recent_cpu_data(cloudwatch_client):
    """
    Fetch all available CPU utilization data from CloudWatch
    
    Args:
        cloudwatch_client: boto3 CloudWatch client
        
    Returns:
        numpy array of CPU utilization values
    """
    try:
        print("\nFetching CPU data from CloudWatch...")
        response = cloudwatch_client.get_metric_data(
            MetricDataQueries=[
                {
                    'Id': 'cpu_util',
                    'MetricStat': {
                        'Metric': {
                            'Namespace': 'AWS/EC2',
                            'MetricName': 'CPUUtilization',
                        },
                        'Period': 300,
                        'Stat': 'Average'
                    },
                    'ReturnData': True
                }
            ],
            StartTime=datetime.now(timezone.utc) - timedelta(days=455),  # ~15 months
            EndTime=datetime.now(timezone.utc)
        )
        
        print("\n=== CloudWatch Data Response ===")
        print(json.dumps(response, indent=2, default=str))
        print("===============================\n")
        
        if response['MetricDataResults']:
            values = response['MetricDataResults'][0]['Values']
            if values:
                print(f"Successfully retrieved {len(values)} CPU utilization values")
                return np.array(values)
            else:
                print("No CPU utilization values found in response")
                return None
        print("No MetricDataResults found in response")
        return None
        
    except Exception as e:
        print(f"\nERROR fetching CloudWatch metrics: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return None

@app.get("/predict")
async def predict():
    try:
        if not aws_credentials["access_key"] or not aws_credentials["secret_key"]:
            return {"success": False, "error": "AWS credentials not set"}
            
        cloudwatch = boto3.client(
            'cloudwatch',
            aws_access_key_id=aws_credentials["access_key"],
            aws_secret_access_key=aws_credentials["secret_key"]
        )
        
        # Get recent CPU data
        recent_values = get_recent_cpu_data(cloudwatch)
        if not recent_values:
            return {"success": False, "error": "No recent CPU data available"}
        
        # Load predictor
        predictor = LSTMPredictor()
        if os.path.exists('models/lstm_model.h5'):
            predictor.model = tf.keras.models.load_model('models/lstm_model.h5')
            predictor.scaler = joblib.load('models/scaler.save')
        else:
            return {"success": False, "error": "Model not trained yet"}
        
        # Make prediction
        predicted_cpu = predictor.predict_next(recent_values)
        need_new_instance = predicted_cpu > 80.0  # Threshold for new instance
        
        return {
            "success": True,
            "need_new_instance": need_new_instance,
            "predicted_max_cpu": predicted_cpu
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/launch-instance")
async def launch_instance(ami_id: str = Form(...), instance_type: str = Form(...)):
    try:
        ec2 = boto3.client(
            'ec2',
            aws_access_key_id=aws_credentials["access_key"],
            aws_secret_access_key=aws_credentials["secret_key"]
        )

        print("\n=== Launching EC2 Instance ===")
        print(f"AMI ID: {ami_id}")
        print(f"Instance Type: {instance_type}")
        
        response = ec2.run_instances(
            ImageId=ami_id,
            InstanceType=instance_type,
            MinCount=1,
            MaxCount=1
        )
        
        print("\n=== EC2 Launch Response ===")
        print(json.dumps(response, indent=2, default=str))
        print("==========================\n")

        instance_id = response['Instances'][0]['InstanceId']
        print(f"Successfully launched instance: {instance_id}")
        
        return {
            "success": True,
            "instance_id": instance_id
        }
    except Exception as e:
        print("\n=== EC2 Launch ERROR ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("=====================\n")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
