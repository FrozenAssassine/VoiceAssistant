import pyttsx3;
import time
engine = pyttsx3.init();

voices = engine.getProperty('voices')
# for voice in voices:
#     print(voice)
#     if voice.name == 'German Male':
#         engine.setProperty('voice', voice.id)

engine.setProperty('rate', 160)

def speakText(text):
# Ändern Sie die Sprache auf Deutsch
    engine.setProperty('voice', 'german')
    engine.say(text);
    engine.runAndWait();

def speakPause(delay):
    time.sleep(int(delay / 10))