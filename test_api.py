import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Get API key from environment
api_key = os.getenv('OWLRACLE_API_KEY')

# Test API endpoint
network = 'eth'
url = f'https://api.owlracle.info/v4/{network}/gas'
params = {'apikey': api_key}

try:
    res = requests.get(url, params=params)
    res.raise_for_status()
    data = res.json()
    print("API Response:")
    print(data)
except requests.exceptions.RequestException as e:
    print(f"Error: {str(e)}") 