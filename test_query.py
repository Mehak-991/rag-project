import requests

response = requests.post(
    'http://localhost:8000/query',
    json={
        'query': 'What is machine learning?',
        'top_k': 5,
        'rerank': True
    }
)

print("Status:", response.status_code)
print("Response:", response.json())
