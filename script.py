import speech_recognition as sr
import pyttsx3
import time
from openai import OpenAI
from config import OPEN_AI_API_KEY
import numpy as np
import soundfile as sf
import os

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

def detect_flush(audio_data, client):
    # Save the audio data temporarily
    temp_path = "temp_audio.wav"
    sf.write(temp_path, np.frombuffer(audio_data.frame_data, dtype=np.int16), audio_data.sample_rate)
    
    try:
        # Open and send the audio file to OpenAI
        with open(temp_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                prompt="If you hear a toilet flushing sound, respond with 'true'. For any other sound, respond with 'false'.",
                response_format="text"
            )
        
        # Convert response to boolean
        return response.strip().lower() == "true"
        
    except Exception as e:
        print(f"Error in audio analysis: {e}")
        return False
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

def listen_and_respond():
    recognizer, speaker, client = initialize_speech()
    
    while True:
        try:
            # Use the microphone as source
            with sr.Microphone() as source:
                print("Listening...")
                # Listen for audio input without noise adjustment
                audio = recognizer.listen(source)
                
                # Check for flush sound
                if detect_flush(audio, client):
                    celebration = "Hooray, that was a good one!"
                    print(f"AI responds: {celebration}")
                    speaker.say(celebration)
                    speaker.runAndWait()
                    continue
                
                # Normal speech processing continues if not a flush
                text = recognizer.recognize_google(audio).lower()
                print(f"You said: {text}")
                
                # Get AI response
                ai_response = get_ai_response(client, text)
                print(f"AI responds: {ai_response}")
                
                # Speak the response
                speaker.say(ai_response)
                speaker.runAndWait()
                    
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