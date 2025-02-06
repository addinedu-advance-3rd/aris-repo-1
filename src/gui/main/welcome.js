import { NGROK_BASE_URL } from './config.js';


document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('nicknameForm');
    const nicknameInput = document.getElementById('nicknameInput');
    const statusMessage = document.getElementById('statusMessage');

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const nickname = nicknameInput.value.trim();
      if (nickname) {
        try {
          // API 요청
          const response = await fetch(`${NGROK_BASE_URL}/face/submit_name`, {
            method: 'POST',
            mode: 'cors',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ name: nickname }),
          });

          const result = await response.json();
          if (response.ok) {
            localStorage.setItem('customerNickname', nickname); // 닉네임 저장
            statusMessage.textContent = result.message;
            if (NGROK_BASE_URL === '') {
              window.location.href = '/gui/topping.html'; // 토핑 선택 페이지로 이동
            } else {
              window.location.href = '/topping.html'; // 토핑 선택 페이지로 이동
            }
          } else {
            statusMessage.textContent = result.error || 'Error occurred.';
          }
        } catch (error) {
          console.error('Error submitting nickname:', error);
          statusMessage.textContent = 'Failed to submit. Please try again.';
        }
      } else {
        statusMessage.textContent = 'Please enter a nickname.';
      }
    });
  });
