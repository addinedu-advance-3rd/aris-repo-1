// TTS 실행 함수 (콜백 없음, 단순 음성 출력)
export function speak(text) {
    const synth = window.speechSynthesis;
    const utterance = new SpeechSynthesisUtterance(text);

    utterance.lang = "ko-KR"; // 한국어 설정
    utterance.rate = 1.0; // 속도 조절
    utterance.pitch = 1.0; // 음높이 조절


    synth.speak(utterance);
}

window.speak = speak;