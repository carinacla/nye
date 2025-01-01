import speech_recognition as sr
import pyttsx3
import time
from openai import OpenAI
from config import OPEN_AI_API_KEY

def initialize_speech():
    # Initialize the recognizer, text-to-speech engine, and OpenAI client
    recognizer = sr.Recognizer()
    speaker = pyttsx3.init()
    client = OpenAI(api_key=OPEN_AI_API_KEY)
    return recognizer, speaker, client

def get_ai_response(client, text):
    # Send text to OpenAI and get response
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant engaged in a verbal conversation. Keep responses concise and natural, as they will be spoken aloud."},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content

def listen_and_respond():
    recognizer, speaker, client = initialize_speech()
    
    while True:
        try:
            # Use the microphone as source
            with sr.Microphone() as source:
                print("Listening...")
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                # Listen for audio input
                audio = recognizer.listen(source)
                
                # Convert speech to text
                text = recognizer.recognize_google(audio).lower()
                print(f"You said: {text}")
                
                # Get AI response
                ai_response = get_ai_response(client, text)
                print(f"AI responds: {ai_response}")
                
                # Speak the response
                # speaker.say(ai_response)
                # speaker.runAndWait()
                    
        except sr.UnknownValueError:
            # Speech was unclear
            continue
        except sr.RequestError as e:
            # API error
            print(f"Could not request results; {e}")
        except Exception as e:
            # Other errors
            print(f"Error: {e}")
            
if __name__ == "__main__":
    listen_and_respond()