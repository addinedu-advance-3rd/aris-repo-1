function playTTS() {
    const audio = new Audio("http://YOUR_STREAMLIT_SERVER_URL/welcome.mp3");
    audio.play();
}

document.addEventListener('DOMContentLoaded', function() {
    playTTS();
});
