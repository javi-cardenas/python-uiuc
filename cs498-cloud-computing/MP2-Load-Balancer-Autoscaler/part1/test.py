import requests

url = "http://127.0.0.1:5000" # localhost testing

data = {"seed": 99}
response = requests.post(url, json=data)
print(response.status_code, response.json())

data = {"num": "25"}
response = requests.post(url, json=data)
print(response.status_code, response.json())

data = {"num": 42}
response = requests.post(url, json=data)
print(response.status_code, response.json())