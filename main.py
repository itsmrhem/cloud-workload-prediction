from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import boto3
from datetime import datetime, timedelta
import json

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
        ec2.describe_regions()
        return {"success": True, "message": "Credentials are valid"}
    except Exception as e:
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
            StartTime=datetime.utcnow() - timedelta(hours=1),
            EndTime=datetime.utcnow()
        )

        return {"success": True, "data": response['MetricDataResults']}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/predict")
async def predict():
    try:
        return {
            "success": True,
            "need_new_instance": True,
            "predicted_max_cpu": 85.5
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

        response = ec2.run_instances(
            ImageId=ami_id,
            InstanceType=instance_type,
            MinCount=1,
            MaxCount=1
        )

        return {
            "success": True,
            "instance_id": response['Instances'][0]['InstanceId']
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
