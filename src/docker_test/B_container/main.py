import socket

def start_server():
    host = '0.0.0.0'  # 모든 IP에서 연결을 수락
    port = 12345       # A와 연결할 포트

    # TCP/IP 소켓 생성
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        # 서버 소켓을 포트에 바인딩
        server_socket.bind((host, port))
        print(f"Server started, waiting for connection on {host}:{port}...")

        # 클라이언트 연결을 기다림 (최대 1개의 클라이언트까지)
        server_socket.listen(1)

        # 클라이언트 연결을 받아들임
        client_socket, client_address = server_socket.accept()
        with client_socket:
            print(f"Connection established with {client_address}")

            # 클라이언트로부터 계속해서 데이터를 받음
            while True:
                data = client_socket.recv(1024)
                if not data:
                    print(f"No_data")
                print(f"Received message: {data.decode()}")

            print("Connection closed")

if __name__ == "__main__":
    start_server()
