import requests
import time

def notify_cup_detect():
    """cup_detect의 Flask API에 요청 전송."""
    try:
        response = requests.post("http://localhost:6001/run_cup_test")
        if response.status_code == 200:
            print("Cup test triggered successfully.")
        else:
            print(f"Failed to trigger cup test. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error communicating with cup_detect: {e}")

if __name__ == "__main__":
    # 5초 동안 카운트다운
    for i in range(10, 0, -1):
        print(f"{i}...", end="", flush=True)
        time.sleep(1)

    # 카운트 완료 후 메시지 출력
    print("\n완성되었습니다.")
    
    # API 호출
    notify_cup_detect()
