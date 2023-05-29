import speech_recognition as sr
import chatbot as chat
import pyttsx3;

engine = pyttsx3.init()
engine.setProperty('rate', 160)

def getInputData(speaktext = ""):
    if len(speaktext) > 0:
        speak.speakText(speaktext)

    audio = r.listen(source)
    return r.recognize_google(audio, language="de-DE").lower()         

def speakText(text):
    engine.setProperty('voice', 'german')
    engine.say(text)
    engine.runAndWait()

r = sr.Recognizer()
with sr.Microphone() as source:
    while True:
        try:            
            text = getInputData()
            print(text)
            if "werner" in text:
                print("Werner hört zu!")
                response = chat.chat(text[text.index("werner") + 6:])
                print(f"> {response}")
                speakText(response)

        except sr.UnknownValueError:
            print("Could not understand audio!")
        except sr.RequestError as e:
            print("Could not access Google speech recognition service; {0}".format(e))


