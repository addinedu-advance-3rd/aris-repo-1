import sqlite3
import json
from flask import Flask, request, jsonify
import numpy as np
import base64
import pickle  # embedding 데이터를 BLOB으로 저장하기 위한 라이브러리

app = Flask(__name__)

DATABASE = '/data/db/mydatabase.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with app.app_context():
        db = get_db()
        db.execute('''
            CREATE TABLE IF NOT EXISTS users (
                u_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER NOT NULL,
                gender TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
        ''')
        db.commit()

# 앱 실행 시 DB 초기화
init_db()

# POST: 사용자 추가
@app.route('/users', methods=['POST'])
def add_user():
    data = request.get_json()

    if not data or not data.get('name') or not data.get('age') or not data.get('gender') or not data.get('embedding'):
        return jsonify({"error": "Name, age, gender, and embedding are required"}), 400

    name = data['name']
    age = data['age']
    gender = data['gender']
    embedding = data['embedding']  # embedding은 리스트로 받거나 base64로 인코딩된 문자열일 수 있음

    # 만약 embedding이 리스트라면, 이를 직렬화하여 저장
    embedding_serialized = pickle.dumps(embedding)  # 직렬화 (이진 데이터로 변환)

    try:
        db = get_db()
        db.execute('INSERT INTO users (name, age, gender, embedding) VALUES (?, ?, ?, ?)', 
                   (name, age, gender, embedding_serialized))
        db.commit()
        return jsonify({"message": "User added successfully!"}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "Email or user already exists"}), 400

# GET: 모든 사용자 조회
@app.route('/users', methods=['GET'])
def get_users():
    db = get_db()
    cursor = db.execute('SELECT u_id, name, age, gender, embedding FROM users')
    users = cursor.fetchall()
    
    user_list = []
    for user in users:
        # embedding을 직렬화된 BLOB 형태로 저장하므로 이를 복원하여 사용
        embedding = pickle.loads(user['embedding'])  # 직렬화된 embedding을 복원
        user_list.append({
            'u_id': user['u_id'],
            'name': user['name'],
            'age': user['age'],
            'gender': user['gender'],
            'embedding': embedding  # 복원된 embedding 값
        })
    
    return jsonify(user_list)

# GET: 특정 사용자 조회
@app.route('/users/<int:u_id>', methods=['GET'])
def get_user(u_id):
    db = get_db()
    cursor = db.execute('SELECT u_id, name, age, gender, embedding FROM users WHERE u_id = ?', (u_id,))
    user = cursor.fetchone()
    
    if user is None:
        return jsonify({"error": "User not found"}), 404

    embedding = pickle.loads(user['embedding'])  # 직렬화된 embedding을 복원
    return jsonify({
        'u_id': user['u_id'],
        'name': user['name'],
        'age': user['age'],
        'gender': user['gender'],
        'embedding': embedding
    })

# DELETE: 특정 사용자 삭제
@app.route('/users/<int:u_id>', methods=['DELETE'])
def delete_user(u_id):
    db = get_db()
    cursor = db.execute('DELETE FROM users WHERE u_id = ?', (u_id,))
    db.commit()
    
    if cursor.rowcount == 0:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"message": "User deleted successfully!"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
