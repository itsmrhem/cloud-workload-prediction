import csv
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
import subprocess

app = FastAPI()

region = "eu-north-1"

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
            aws_secret_access_key=aws_credentials["secret_key"],
            region_name=region
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
            aws_secret_access_key=aws_credentials["secret_key"],
            region_name=region

        )

        start_time = datetime.now(timezone.utc) - timedelta(days=455) 
        response = cloudwatch.get_metric_data(
            MetricDataQueries=[
                {
                    'Id': 'cpu_util',
                    'MetricStat': {
                        'Metric': {
                            'Namespace': 'AWS/EC2',
                            'MetricName': 'CPUUtilization',
                        },
                        'Period': 1,
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

        # Convert to CSV and save
        if response['MetricDataResults'] and response['MetricDataResults'][0]['Values']:
            with open('predict/cloudwatch.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['timestamp', 'cpu_usage'])
                
                timestamps = response['MetricDataResults'][0]['Timestamps']
                values = response['MetricDataResults'][0]['Values']
                
                # Sort by timestamp
                data_pairs = sorted(zip(timestamps, values), key=lambda x: x[0])
                
                for timestamp, value in data_pairs:
                    # Convert timestamp to desired format without timezone info
                    formatted_timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')
                    writer.writerow([formatted_timestamp, value])

        return {"success": True, "data": response['MetricDataResults']}
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_recent_cpu_data(cloudwatch_client):
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
                        'Period': 240,
                        'Stat': 'Average'
                    },
                    'ReturnData': True
                }
            ],
            StartTime=datetime.now(timezone.utc) - timedelta(days=455),  
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
        result = subprocess.run(['python', 'models/lstm_predictor_percent.py'], 
                              capture_output=True, 
                              text=True)
        
        if result.returncode != 0:
            return {"success": False, "error": f"Prediction failed: {result.stderr}"}
            
        try:
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines:
                if line.startswith("Next 15 minutes prediction:"):
                    predicted_cpu = float(line.split(":")[1].strip())
                    need_new_instance = predicted_cpu > 80.0
                    
                    return {
                        "success": True,
                        "need_new_instance": need_new_instance,
                        "predicted_max_cpu": predicted_cpu
                    }
            return {"success": False, "error": "Could not find prediction in output"}
        except ValueError as e:
            return {"success": False, "error": f"Failed to parse prediction: {str(e)}"}
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
