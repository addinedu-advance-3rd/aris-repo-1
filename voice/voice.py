import streamlit as st
from gtts import gTTS
import os

st.title("TTS 음성 생성")

message = "어서오세요, 아이스크림 가게입니다!"
if st.button("음성 생성"):
    tts = gTTS(text=message, lang='ko')
    tts.save("welcome.mp3")
    st.audio("welcome.mp3", format="audio/mp3")
