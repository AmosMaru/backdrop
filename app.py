from flask import Flask, render_template, request, jsonify
from flask import Flask, render_template, request, jsonify
from pathlib import Path
from typing import List, Tuple

from langchain import PromptTemplate, LLMChain
from langchain.document_loaders import TextLoader
from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms import GPT4All
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from pydantic import BaseModel, Field
from langchain.chains import ConversationalRetrievalChain
import os
# Constants
from dotenv import load_dotenv
from flask import Flask, redirect, render_template, request, Markup
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
# import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification modelc:\Users\amosm\Desktop\Project\Mche-chatbot\utils
# load .env file
dotenv_path = '.env'  # Assuming the .env file is in the same directory as your script
load_dotenv(dotenv_path)



llama_embeddings_model = os.getenv("LLAMA_EMBEDDINGS_MODEL")
model_type = os.getenv("MODEL_TYPE")
model_path = os.getenv("MODEL_PATH")
model_n_ctx = os.getenv("MODEL_N_CTX")
openai_api_key = os.getenv("OPEN_AI_API_KEY")
index_path = os.getenv("INDEX_PATH")
text_path = os.getenv("TEXT_PATH")

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'models2\plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()



# Functions
def initialize_embeddings() -> LlamaCppEmbeddings:
    return LlamaCppEmbeddings(model_path=llama_embeddings_model)

def load_documents() -> List:
    loader = TextLoader(text_path)
    return loader.load()

def split_chunks(sources: List) -> List:
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=32)
    for chunk in splitter.split_documents(sources):
        chunks.append(chunk)
    return chunks

def generate_index(chunks: List, embeddings: LlamaCppEmbeddings) -> FAISS:
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    return FAISS.from_texts(texts, embeddings, metadatas=metadatas)

# Main execution
# llm = GPT4All(model=local_path, n_ctx=2048, verbose=True)
llm = OpenAI(openai_api_key=openai_api_key)

embeddings = initialize_embeddings()
# sources = load_documents()
# chunks = split_chunks(sources)
# vectorstore = generate_index(chunks, embeddings)
# vectorstore.save_local("full_sotu_index")



from langchain.prompts import PromptTemplate

index = FAISS.load_local(index_path, embeddings)

qa = ConversationalRetrievalChain.from_llm(llm, index.as_retriever(), max_tokens_limit=400)




def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/get', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        if 'msg' in request.form:
            msg = request.form["msg"]
            input_text = msg
            # Replace the following line with your function that generates the chat response
            chat_response = get_Chat_response(input_text)
            return chat_response
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                # You can process the image file here (e.g., using a machine learning model)
                # For now, let's just return a message
                return 'Image uploaded. Predictions will be printed here.'
    return render_template('chat.html')  # Render chat.html when accessing /chat without a form submission

 
def get_Chat_response(query):
    chat_history = []


    # template = (
    # """You are an advisory chatbot call Mche-GPT and YOU Answer Question based on the context 
    # Context: {context}
    # ---
    # Question: {question}
    # Answer: Let's think step by step."""
    # )
    # prompt = PromptTemplate.from_template(template)
    # question_generator_chain = LLMChain(llm=llm, prompt=prompt)
    # qa = ConversationalRetrievalChain(
    # retriever=index.as_retriever(),
    # question_generator=question_generator_chain,
    #  max_tokens_limit=400
    #  )
    
    result = qa({"question": query, "chat_history": chat_history})
    

    return result['answer']


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Mche - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)

if __name__ == '__main__':
    app.run()