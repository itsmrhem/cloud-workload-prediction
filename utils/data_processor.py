import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import boto3

def get_recent_cpu_data(cloudwatch_client, hours=1):
    """Fetch recent CPU utilization data from CloudWatch"""
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
        StartTime=datetime.utcnow() - timedelta(hours=hours),
        EndTime=datetime.utcnow()
    )
    
    if response['MetricDataResults']:
        values = response['MetricDataResults'][0]['Values']
        return values if values else None
    return None
