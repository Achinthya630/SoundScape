import pathlib
import google.generativeai as genai
import PIL.Image
import cv2
import numpy as np
from dotenv import load_dotenv
import os
from googletrans import Translator

class GeminiVision:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
            
        genai.configure(api_key=api_key)
        # Update the model name to use gemini-1.5-flash
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def get_description(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = PIL.Image.fromarray(rgb_frame)
        
        try:
            response = self.model.generate_content([
                # "Describe this image in detail for a visually impaired person. Include important objects, people, actions, and the overall scene context.",
                "Describe the scene concisely, focusing only on important objects, obstacles, and contextual details useful for navigation. Ignore colors, clothing details, and unnecessary descriptions. Use the following structure:Objects: List important objects.Obstacles: Mention any obstacles in the path.Context: Describe the surroundings briefly.Navigation Hints: If relevant, provide movement guidance such as the approximate distance to some obstacle.Do not give answers in points, but explain the context as a normal conversation.",
                pil_image
            ])
            return response.text
        except Exception as e:
            print(f"Error generating description: {e}")
            return "I encountered an error while analyzing the image."
    
    def analyze_brightness(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = PIL.Image.fromarray(rgb_frame)
        
        try:
            response = self.model.generate_content([
                "Analyze the brightness of this image. Is it bright, dim, or dark? Consider natural and artificial lighting.",
                pil_image
            ])
            return response.text
        except Exception as e:
            print(f"Error analyzing brightness: {e}")
            return "I encountered an error while analyzing the brightness."
    
    def read_text(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = PIL.Image.fromarray(rgb_frame)
        
        try:
            response = self.model.generate_content([
                "Read and extract any visible text in this image. Format it clearly.Mention the location of text if relevant.",
                pil_image
            ])
            return response.text
        except Exception as e:
            print(f"Error reading text: {e}")
            return "I encountered an error while reading the text."
    
    def analyze_form(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = PIL.Image.fromarray(rgb_frame)
        
        try:
            response = self.model.generate_content([
                "This is a form or document. Identify and read all fields, labels, and entered information. Format it as field: value pairs. Help me fill the form by providing me the details.",
                pil_image
            ])
            return response.text
        except Exception as e:
            print(f"Error analyzing form: {e}")
            return "I encountered an error while analyzing the form."







# import pathlib
# import google.generativeai as genai
# import PIL.Image
# import cv2
# import numpy as np
# from dotenv import load_dotenv
# import os

# class GeminiVision:
#     def __init__(self):
#         # Load environment variables from .env file
#         load_dotenv()
#         api_key = os.getenv('GEMINI_API_KEY')
#         if not api_key:
#             raise ValueError("GEMINI_API_KEY not found in .env file")
            
#         genai.configure(api_key=api_key)
#         self.model = genai.GenerativeModel('gemini-pro-vision')
    
#     def get_description(self, frame):
#         # Convert OpenCV BGR image to RGB
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # Convert numpy array to PIL Image
#         pil_image = PIL.Image.fromarray(rgb_frame)
        
#         # Generate description using Gemini
#         response = self.model.generate_content([
#             "Describe this image in detail for a visually impaired person. Include important objects, people, actions, and the overall scene context.",
#             pil_image
#         ])
        
#         return response.text