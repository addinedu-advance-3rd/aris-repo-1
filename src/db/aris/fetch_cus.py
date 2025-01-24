import requests

def fetch_cus():
    url = 'http://127.0.0.1:8000/api/Customers/'
    response = requests.get(url)

    if response.status_code == 200:
        Customers = response.json()  # JSON 응답을 파싱
        for cus in Customers:
            print(f"ID: {cus['id']}, Name: {cus['name']}, Age: {cus['age']}, Gender: {cus['gender']}")
    else:
        print(f"Failed to retrieve items. Status code: {response.status_code}")

