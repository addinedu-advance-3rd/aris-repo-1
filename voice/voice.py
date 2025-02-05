import streamlit as st
from gtts import gTTS
import os

st.title("TTS 음성 생성")

message = "어서오세요, 아이스크림 가게입니다!"
topping = "토핑을 골라주세요"

if st.button("환영메세지"):
    tts = gTTS(text=message, lang='ko')
    tts.save("welcome.mp3")
    st.audio("welcome.mp3", format="audio/mp3")

if st.button("토핑메세지"):
    tts2= gTTS(text=topping, lang='ko')
    tts2.save("topping.mp3")
    st.audio("topping.mp3", format="audio/mp3")

