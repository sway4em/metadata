# requirements.txt
# import boto3
# import botocore
# import llama-index
# import langchain
# import streamlit
# import pypdf
# main.py

#####################################################################
# Amazon Bedrock - boto3
#####################################################################

import boto3

# Setup bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2",
)

#####################################################################
# LLM - Amazon Bedrock LLM using LangChain
#####################################################################

from llama_index.llms.langchain import LangChainLLM
from langchain.llms import Bedrock

model_id = "anthropic.claude-v2"
model_kwargs =  { 
    "max_tokens_to_sample": 4096,
    "temperature": 0.5,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman:"],
}

llm = Bedrock(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs=model_kwargs
)

#####################################################################
# Embedding Model - Amazon Titan Embeddings Model using LangChain
#####################################################################

from llama_index.embeddings.langchain import LangchainEmbedding
# Source code - https://github.com/run-llama/llama_index/blob/main/llama_index/embeddings/__init__.py
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.embeddings.langchain import LangchainEmbedding

from langchain.embeddings import BedrockEmbeddings

# create embeddings
bedrock_embedding = BedrockEmbeddings(
    client=bedrock_runtime,
    model_id="amazon.titan-embed-text-v1",
)

# load in Bedrock embedding model from langchain
embed_model = LangchainEmbedding(bedrock_embedding)

#####################################################################
# Service Context
#####################################################################

from llama_index.core import ServiceContext, set_global_service_context

service_context = ServiceContext.from_defaults(
  llm=llm,
  embed_model=embed_model,
  system_prompt="You are an AI assistant answering questions."
)

set_global_service_context(service_context)

#####################################################################
# Streamlit
#####################################################################

import streamlit as st
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex

st.set_page_config(
  page_title="LlamaIndex Q&A over you data 📂",
  page_icon="🦙",
  layout="centered",
  initial_sidebar_state="auto",
  menu_items=None)

st.title("LlamaIndex 🦙 Q&A over your data 📂")

@st.cache_resource(show_spinner=False)
def load_data():
  """
    Loads and indexes the data using the VectorStoreIndex.
    
    Returns:
    - VectorStoreIndex: Indexed representation of your data.
  """
  with st.spinner(
    text="Loading and indexing your data. This may take a while..."):
    reader=SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs=reader.load_data()

    index=VectorStoreIndex.from_documents(docs)
    return index

# Create Index
index=load_data()

# Create Query Engine
query_engine=index.as_query_engine(similarity_top_k=3)

# Take input from the user
user_input=st.text_input("Enter Your Query", "")

# Display the input
if st.button("Submit"):
  st.write(f"Your Query: {user_input}")

  with st.spinner("Thinking..."):
    # Query the index
    result=query_engine.query(f"\n\nHuman:{user_input}\n\nAssistant:")

    # Display the results
    st.write(f"Answer: {str(result)}")