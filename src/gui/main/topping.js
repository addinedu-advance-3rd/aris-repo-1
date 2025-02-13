import { NGROK_BASE_URL } from './config.js';
import { speak } from './tts.js';

document.addEventListener('DOMContentLoaded', function() {
    const bgContainer = document.querySelector('.bg-container');
    const rowCount = 22;
    const itemsPerRow = 22;
    const horizontalSpacing = 150; // px
    const verticalSpacing = 100;   // px
    
    let html = '';
    for (let row = 0; row < rowCount; row++) {
      for (let col = 0; col < itemsPerRow; col++) {
        const left = col * horizontalSpacing;
        const top = row * verticalSpacing;
        const imageIndex = (col % 3) + 1; // 순차적으로 1, 2, 3번 이미지 사용
        html += `
          <div class="item" style="top: ${top - 500}px; left: ${left - 700}px;">
            <img src="img_src/ice_img_${imageIndex}.png" alt="Ice Cream">
          </div>
        `;
      }
    }
    bgContainer.innerHTML = html;
  });


document.addEventListener('DOMContentLoaded', () => {
    const nicknameDisplay = document.getElementById('nickname-display');
    const form = document.getElementById('topping-form');

    // (1) 닉네임 불러오기
    const nickname = localStorage.getItem('customerNickname');
    if (nickname) {
      nicknameDisplay.textContent = `@${nickname} 님 반가워요!`;
      speak(`@${nickname} 님, 반가워요! 토핑을 선택해주세요.`);
    } else {
      nicknameDisplay.textContent = '고객님 반가워요!';
      speak(`토핑을 선택해주세요.`);
    }

    // (추가 기능) 토핑 추천 하기 나이, 성별 가져오기
    const age = localStorage.getItem('customerAge')
    const gender = localStorage.getItem('customerGender')
    const recommendationDisplay = document.getElementById('recommendation-display');
    
    if (age && gender) {
        const recommendedToppings = getRecommendedToppings(parseInt(age), gender);
        recommendationDisplay.textContent = `추천 토핑: ${recommendedToppings.join(', ')}`;
    } else {
        recommendationDisplay.textContent = '추천 정보를 가져오지 못했습니다.';
    }

    // 토핑 추천 로직
    function getRecommendedToppings(age, gender) {
      const allToppings = ["조리퐁", "해바라기씨", "코코볼"]; // 전체 토핑 리스트
      let recommendedToppings = [];
  
      // 💡 연령대 & 성별에 따른 기본 추천 (1개 이상)
      if (age <= 19) {
          recommendedToppings.push(gender === "male" ? "조리퐁" : "코코볼");
      } else if (age <= 29) {
          recommendedToppings.push("조리퐁");
          recommendedToppings.push(gender === "male" ? "코코볼" : "해바라기씨");
      } else if (age <= 39) {
          recommendedToppings.push("코코볼", "해바라기씨"); // 30대부터 해바라기씨 추천 증가
      } else {
          recommendedToppings.push("해바라기씨"); // 40대 이상이면 해바라기씨 기본 추천
      }
  
      // ✅ 무작위로 추천 개수 결정 (1~3개)
      const targetToppingCount = Math.floor(Math.random() * 3) + 1; // 1~3개 중 랜덤 선택
  
      // ✅ 부족하면 랜덤 추가 (이미 추천된 항목 제외)
      while (recommendedToppings.length < targetToppingCount) {
          let randomTopping = allToppings[Math.floor(Math.random() * allToppings.length)];
          if (!recommendedToppings.includes(randomTopping)) { // 중복 방지
              recommendedToppings.push(randomTopping);
          }
      }
  
      return recommendedToppings;
  }
    
    // (2) 폼 제출 시 이벤트 핸들러
    form.addEventListener('submit', async (event) => {
      event.preventDefault();

      // 체크된 토핑 정보 수집
      const selectedToppings = Array.from(
        document.querySelectorAll('input[name="topping"]:checked')
      ).map((checkbox) => checkbox.value);

      console.log('Selected Toppings:', selectedToppings);


      // (2) True/False로 변환
      // Chocolate, Sprinkles, Strawberries 순서로 매핑
      const toppingOrder = ["Chocolate", "Sprinkles", "Strawberries"];
      const booleanToppings = toppingOrder.map((topping) =>
        selectedToppings.includes(topping)
      );
      console.log("booleanToppings:", booleanToppings);


      // (3) control_service (8080)로 POST 전송
      try {
        const response = await fetch(`${NGROK_BASE_URL}/control_service/select_toppings`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ toppings: booleanToppings }),
        });

        if (!response.ok) {
          // 응답이 200~299 범위가 아니면 에러 처리
          const errorData = await response.json();
          console.error('Server Error:', errorData);
          alert(errorData.error || '서버 에러가 발생했습니다.');
          return;
        }

        // 서버에서 온 응답 결과 확인
        const result = await response.json();
        console.log('Server response:', result);

        // TODO: 주문 완료 후 로직 (DB 저장, 알림 등)
        alert('주문이 완료되었습니다!');

        // (4) 다음 페이지로 이동 (Webcam Video Recorder 등)
        window.location.href = '/gui/memory.html';
      } catch (error) {
        console.error('Error submitting toppings:', error);
        alert('서버와 통신 중 에러가 발생했습니다.');
      }
    });
  });