import socket
import time

def send_message_to_server():
    host = '172.28.0.3'  # B의 고정 IP 주소
    port = 12345          # B의 서버 포트

    # TCP 소켓 생성
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        try:
            # B에 연결
            client_socket.connect((host, port))
            print(f"Connected to server at {host}:{port}")

            # 메시지 계속해서 전송
            count = 1
            while True:
                message = f"Message {count}"
                client_socket.sendall(message.encode())
                print(f"Sent: {message}")
                count += 1
                time.sleep(1)  # 1초마다 메시지 전송

        except Exception as e:
            print(f"Error connecting to server: {e}")

if __name__ == "__main__":
    send_message_to_server()
