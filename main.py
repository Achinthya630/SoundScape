import functions
import speech
import cv2
import os
import detect
import datetime
from gemini_vision import GeminiVision
# frontend
from gui import main

if __name__ == '__main__':
    main()

# frontend end

# Initialize speech engine and other components
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= "api-key.json"
project_id = "project_id"
engine = speech.speech_to_text()

# Initialize Gemini Vision
gemini = GeminiVision()

listening = False
intent = None

while True:
    cam = cv2.VideoCapture(0)
    if not listening:
        resp = engine.recognize_speech_from_mic()
        print(resp)
        if(resp != None):
            intent, text = detect.detect_intent_texts(project_id, 0, [resp], 'en')
        if(intent == 'Hello' and resp!=None):
            listening = True

    else:
        engine.text_speech("What can I help you with?")
        intent = ''
        engine.text_speech("Listening")
        resp = engine.recognize_speech_from_mic()
        if resp is None:
            print("No speech detected. Please try again.")
            engine.text_speech("I didn't catch that. Can you repeat?")
            continue
        
        engine.text_speech("Processing")
        if(resp!=None):
            print(resp)
            intent, text = detect.detect_intent_texts(project_id, 0, [resp], 'en')
        
        if intent == 'Describe':
            ret, frame = cam.read()
            if ret:
                description = gemini.get_description(frame)
                print("Description:", description)
                engine.text_speech(description)
            else:
                engine.text_speech("Could not capture image from camera")
                
        elif intent == 'Brightness':
            ret, frame = cam.read()
            if ret:
                brightness = gemini.analyze_brightness(frame)
                print("Brightness:", brightness)
                engine.text_speech(brightness)
            else:
                engine.text_speech("Could not capture image from camera")
                
        elif intent == "Read":
            ret, frame = cam.read()
            if ret:
                text_content = gemini.read_text(frame)
                print("Text:", text_content)
                engine.text_speech(text_content)
            else:
                engine.text_speech("Could not capture image from camera")
                
        elif intent == "FillForm":
            ret, frame = cam.read()
            if ret:
                form_content = gemini.analyze_form(frame)
                print("Form:", form_content)
                engine.text_speech(form_content)
            else:
                engine.text_speech("Could not capture image from camera")
                
        elif intent == "Time":
            # currentDT = datetime.datetime.now()
            # engine.text_speech("The time is {} hours and {} minutes".format(currentDT.hour, currentDT.minute))
            currentDT = datetime.datetime.now()
            time_str = f"The time is {currentDT.strftime('%I:%M %p')}"  # Format as 12-hour time with AM/PM
            print("Time:", time_str)
            engine.text_speech(time_str)

        elif intent == 'endconvo':
            print(text)
            listening = False
            engine.text_speech(text)
            
        elif resp != 'None':
            engine.text_speech(text)
            
    cam.release()