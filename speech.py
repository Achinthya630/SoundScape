from google.cloud import texttospeech
import speech_recognition as sr
import pygame
import io
import tempfile
import os
import pyttsx3
from google.oauth2 import service_account

from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import re
import nltk
from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

class speech_to_text:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.tts_client = texttospeech.TextToSpeechClient()
        pygame.mixer.init()
        self.microphone = sr.Microphone()
        self.credentials = service_account.Credentials.from_service_account_file('api-key.json')

    def recognize_speech_from_mic(self):
        print("Start...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)
        print("Found mic")
        response = {
            "success": True,
            "error": None,
            "transcription": None
        }
        try:
            response["transcription"] = self.recognizer.recognize_google(audio)
        except sr.RequestError:
            # API was unreachable or unresponsive
            response["success"] = False
            response["error"] = "API unavailable"
        except sr.UnknownValueError:
            # speech was unintelligible
            response["error"] = "Unable to recognize speech"
        if(response["transcription"]=="None"):
            print("Speech not detected! Pls try again!")
        return response["transcription"]

    def clean(self, text):
        lem = WordNetLemmatizer()
        stem = PorterStemmer()
        stop_words = set(stopwords.words("english"))
        new_words = ["hey", "hi", "hello", "what's up", "i", "please", "help", "using", "show", "result", "large",
                     "also", "iv", "one", "two", "new", "previously", "shown"]
        stop_words = stop_words.union(new_words) - {"whom", "who"}
        text = text.lower()
        text = text.split()
        ps = PorterStemmer()
        lem = WordNetLemmatizer()
        text = [lem.lemmatize(word) for word in text if not word in
                                                            stop_words]
        text = " ".join(text)
        return text

    def text_speech(self, text):
        synthesis_input = texttospeech.SynthesisInput(text=text)

        # Configure the voice
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Neural2-D",  # This is a more natural-sounding voice
            ssml_gender=texttospeech.SsmlVoiceGender.MALE
        )

        # Configure the audio
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,  # Normal speed
            pitch=0,  # Normal pitch
            volume_gain_db=0.0  # Normal volume
        )

        try:
            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )

            # Save the audio to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                temp_file.write(response.audio_content)
                temp_path = temp_file.name

            # Play the audio using pygame
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            # Clean up the temporary file
            pygame.mixer.music.unload()
            os.unlink(temp_path)

        except Exception as e:
            print(f"Error in text-to-speech: {e}")


