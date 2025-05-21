import requests
import json
import random
from datetime import datetime, timedelta
import ipaddress

def generate_random_transaction():
    # Generate random transaction ID
    transaction_id = f"TXN{datetime.now().strftime('%Y%m%d')}{random.randint(100, 999)}"
    
    # Generate random account ID
    account_id = f"ACC{random.randint(100000, 999999)}"
    
    # Generate random date within the last 30 days
    date = datetime.now() - timedelta(days=random.randint(0, 30))
    transaction_date = date.strftime('%d/%m/%Y')
    
    # List of possible transaction types
    transaction_types = ['transfer', 'withdrawal', 'deposit', 'payment']
    
    # List of possible locations
    locations = ['Harare CBD', 'Bulawayo', 'Gweru', 'Mutare', 'Victoria Falls']
    
    # Generate random device ID
    device_id = f"DVC-{random.randint(100000, 999999)}"
    
    # Generate random IP address
    ip = str(ipaddress.IPv4Address(random.randint(0, 2**32-1)))
    
    # Generate random login time
    login_hour = random.randint(0, 23)
    login_minute = random.randint(0, 59)
    login_time = f"{login_hour:02d}:{login_minute:02d}"
    
    # Generate random amount between 10 and 10000
    amount = round(random.uniform(10, 10000), 2)
    
    # Generate random anomaly score between 0 and 1
    anomaly_score = round(random.uniform(0, 1), 2)
    
    # Generate random suspicious flag (0 or 1)
    suspicious_flag = random.randint(0, 1)
    
    # Generate random account balance between 1000 and 100000
    account_balance = round(random.uniform(1000, 100000), 2)
    
    return {
        "transaction_id": transaction_id,
        "account_id": account_id,
        "transaction_date": transaction_date,
        "transaction_type": random.choice(transaction_types),
        "location": random.choice(locations),
        "device_id": device_id,
        "ip_address": ip,
        "login_time": login_time,
        "amount": amount,
        "anomaly_score": anomaly_score,
        "suspicious_flag": suspicious_flag,
        "account_balance": account_balance
    }

def post_transactions(num_transactions=100):
    url = "http://localhost:8000/api/predict-transaction/"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Token c4f64fecc334b1198df957561b12a886a39bc430"
    }
    
    success_count = 0
    for i in range(num_transactions):
        transaction = generate_random_transaction()
        try:
            response = requests.post(url, json=transaction, headers=headers)
            if response.status_code == 200 or response.status_code == 201:
                success_count += 1
                print(f"Transaction {i+1} posted successfully")
            else:
                print(f"Failed to post transaction {i+1}. Status code: {response.status_code}")
        except Exception as e:
            print(f"Error posting transaction {i+1}: {str(e)}")
    
    print(f"\nCompleted: {success_count} out of {num_transactions} transactions posted successfully")

if __name__ == "__main__":
    # Generate and post 100 transactions
    post_transactions(100)