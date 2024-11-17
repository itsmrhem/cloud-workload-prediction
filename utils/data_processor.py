import numpy as np
from datetime import datetime, timedelta

def get_recent_cpu_data(cloudwatch_client, hours=1):
    """
    Fetch recent CPU utilization data from CloudWatch
    
    Args:
        cloudwatch_client: boto3 CloudWatch client
        hours: number of hours of historical data to fetch
        
    Returns:
        numpy array of CPU utilization values
    """
    try:
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
            return np.array(values) if values else None
        return None
        
    except Exception as e:
        print(f"Error fetching CloudWatch metrics: {str(e)}")
        return None
