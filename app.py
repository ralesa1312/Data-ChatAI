
#import packages

import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64


import os
from datetime import datetime


#import langchain

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate



#getting text chunks pdf

def get_pdf_etext(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text

#getting chunks from text

def get_text_chunks(text, model_name):
    if model_name == "Google AI":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

#embedding this chunks and storing them in a vector

def get_vector_store(text_chunks, model_name, api_key=None):
    if model_name == "GooGle Ai":
        embeddings = GoogleGenerativeAIEmbeddings(model="model/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")

    return vector_store

#create a conversational chain using langchain

def get_conversationa_chain(model_name, vector_store=None, api_key=None):
    if model_name == "Google AI":
        propmt_template = """
        Answer the question as detailled as possible from the provided context, make sure to provide all details with proper structure, if 
        the answer is not in the provided context, just say, "answer is not in available in the context", don't provide the wrong answer\n\n
        Context:\n{context}?\n
        Question:\n{context}?\n
        
        Answer:"""

        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
        prompt = PromptTemplate(template=propmt_template, input_variables=['context','questions'])
        chain=load_qa_chain(model, chain_type="stuff",prompt=prompt)
        return chain
    
#take user input

    def user_input(user_question, model_name, api_key, pdf_docs, conversation_history):
        if api_key is None or pdf_docs is None:
            st.warning("Please upload  any pdf and provide api key")
            return
        text_chunks = get_text_chunks(get_pdf_etext(pdf_docs), model_name)
        vector_store = get_vector_store(text_chunks, model_name, api_key)
        user_question_output=""
        response_output=""

        if model_name == "Google AI":
            embeddings = GoogleGenerativeAIEmbeddings(model="model/embedding-001", google_api_key=api_key)
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question)