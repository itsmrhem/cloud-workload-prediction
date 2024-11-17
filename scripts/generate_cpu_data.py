import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate synthetic CPU usage percentage data
def generate_cpu_data():
    # Generate 1000 hourly timestamps
    start_date = datetime.now() - timedelta(days=42)  # About 1000 hours back
    dates = [start_date + timedelta(hours=x) for x in range(1000)]
    
    # Generate synthetic CPU usage percentages
    # Base pattern: Daily cycles with some weekly patterns
    base = np.linspace(0, 4*np.pi, 1000)  # Multiple cycles
    daily_pattern = 30 + 20 * np.sin(base)  # Oscillate between roughly 10-50%
    weekly_pattern = 10 * np.sin(base/7)    # Weekly variation
    
    # Add some random noise
    noise = np.random.normal(0, 5, 1000)
    
    # Combine patterns and ensure values stay between 0-100
    cpu_usage = np.clip(daily_pattern + weekly_pattern + noise, 0, 100)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'cpu_usage': cpu_usage
    })
    
    return df

if __name__ == "__main__":
    df = generate_cpu_data()
    df.to_csv('2.csv', index=False)
    print("Generated CPU usage percentage data saved to 2.csv")
