"""
This module provides a Streamlit-based chatbot interface for question-answering
tasks using the OpenAI language model and the Chroma vectorstore.

Author: Arunkumar M
"""

#Importing the required libraries
import streamlit as st
import os
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# Initialize the OpenAI language model
model = ChatOpenAI(openai_api_key=os.getenv("API_KEY"))

# Initialize the sentence transformer embeddings
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


# Path to the directory for persisting the vectorstore
persist_directory="dbbbb"


#Importing and retrieving the vector database
vectordb1=Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

# Set up the retriever for the vectorstore
retriever = vectordb1.as_retriever()


# Define the prompt template for the chatbot
template = """You are an assistant for question-answering tasks. \
Answer the question based only on the following context: \
Summarize and answer \
Based on the given tweet context give the sentiment of the tweet, name of the stock affected. Give an one line explanation for the reason of affection. \
Give the three answers in seperate lines. \

{context}

Question: {question}
"""

# Create the chat prompt template
prompt = ChatPromptTemplate.from_template(template) #prompt for the OpenAI model


# 1. The chain passes the question to the retriver
# 2. The retriver gets the aprropriate context and passed it to the model(OpenAI)
# 3. The model generates the answer based on the context.

chain = (
    {"context": retriever, "question": RunnablePassthrough()} #retriever
    | prompt
    | model #generater
    | StrOutputParser()
)

#Streamlit UI
#Title for the webpage
st.title("Tweet Bot")

#Text Box for the user to enter their queries
user_ques=st.text_area("Enter the tweet : ")


# Generate the response when the button is clicked
if st.button("Calculate"):
    st.spinner(text="In progress...")
    try:
        result = chain.invoke(user_ques) #passing the user query to the chain for retrieval and generation of answer
        st.write(result) #writing the result back to the user
    except Exception as e:
        st.error(f"An error occurred: {e}")


