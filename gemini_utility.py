import os
import json
from PIL import Image

import google.generativeai as genai

# working directory path
working_dir = os.path.dirname(os.path.abspath(__file__))

# path of config_data file
config_file_path = f"{working_dir}/config.json"
# print(config_file_path )
config_data = json.load(open("config.json"))

# loading the GOOGLE_API_KEY
GOOGLE_API_KEY = config_data["GOOGLE_API_KEY"]
# print(GOOGLE_API_KEY)

# configuring google.generativeai with API key
genai.configure(api_key=GOOGLE_API_KEY)


# function to load gemini-pro-model for chatbot
def load_gemini_pro_model():
    gemini_pro_model = genai.GenerativeModel("gemini-1.5-flash")
    return gemini_pro_model


# get response from Gemini-Pro-Vision model - image/text to text
# def gemini_pro_vision_response(prompt, image):
#     gemini_pro_vision_model = genai.GenerativeModel("gemini-pro-vision")
#     response = gemini_pro_vision_model.generate_content([prompt, image])
#     result = response.text
#     return result
# image = Image.open ("test image.png")
# prompt = "write a short caption for this image"
# output = gemini_ pro_vision response (prompt, image)
# print (output).

# List available models

genai.configure(api_key=GOOGLE_API_KEY)
models = genai.list_models()
for model in models:
    print(f"Model: {model.name}")

#  image/text to text
def load_gemini_flash_model():
    return genai.GenerativeModel("gemini-1.5-flash")

def gemini_flash_vision_response(prompt, image):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")  # Or another model from the list
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        print(f"Error: {e}")
        return "Error generating response."

# get response from embeddings model - text to embeddings
def embeddings_model_response(input_text):
    embedding_model = "models/embedding-001"
    embedding = genai.embed_content(model=embedding_model,
                                    content=input_text,
                                    task_type="retrieval_document")
    embedding_list = embedding["embedding"]
    return embedding_list


# get response from Gemini-Pro model - text to text
# get response from Gemini-Pro model - text to text
def gemini_pro_response(user_prompt):
    try:
        # Use a valid model from the list (e.g., gemini-1.5-flash)
        gemini_pro_model = genai.GenerativeModel("gemini-1.5-flash")  # Update model name here
        response = gemini_pro_model.generate_content(user_prompt)  # Check if the model supports this method
        result = response.text
        return result
    except Exception as e:
        print(f"Error: {e}")
        return "Model not available or error occurred."# result = gemini_pro_response("What is Machine Learning")
# print(result)
# print("-"*50)
#
#
# image = Image.open("test_image.png")
# result = gemini_pro_vision_response("Write a short caption for this image", image)
# print(result)
# print("-"*50)
#
#
# result = embeddings_model_response("Machine Learning is a subset of Artificial Intelligence")
# print(result)
