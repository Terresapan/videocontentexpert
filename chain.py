__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
import streamlit as st


@st.cache_resource(show_spinner=False)
def initialize_chain(_llm, system_prompt, _memory):

    # Load pdf documents
    pdf_files = [
        "./data/Blogs.pdf",
        "./data/Book.pdf",
    # Add more PDF file paths here
    ]

    documents = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(file_path=pdf_file)
        docs_lazy = loader.lazy_load()
        documents.extend(docs_lazy)

    if documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        document_chunks = text_splitter.split_documents(documents)

        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(document_chunks, embeddings)

        qa = ConversationalRetrievalChain.from_llm(_llm, vectorstore.as_retriever(), memory=_memory)
        return qa
    else:
        raise Exception("No PDFs found. Please add files in the specified folder to proceed.")
    

def generate_suggestions(_llm, video_topic, target_audience, selling_point, question):
    """Generates suggestions based on user input."""
 
    system_message = f"""
        You are a highly skilled expert in social media strategies, specializing in the creation of engaging short video content to drive lead generation.
        You will be provided the basic info about the user, including {video_topic},{target_audience},{selling_point}.
        Your role is to answer {question} and assist users in producing effective short videos based on a series of provided PDFs, which serve as your primary source of information.
        
        Your response should include:
        1. A brief outline for a short video (30-60 seconds) that addresses the topic and appeals to the target audience.
        2. Suggestions on how to incorporate the unique selling points into the video.
        3. A response to the user's specific question, relating it back to the video content.
        4. Any relevant tips or strategies from the PDFs that could enhance the video's effectiveness for lead generation.       
                
        Adhere strictly to the content in these PDFs. If the information requested is unrelated to the provided materials,
        clearly state that you do not have the answer rather than providing speculative responses.
        If users raise questions unrelated to content creation, politely remind them to focus on the topic at hand.
    """

    messages = [  
        ("system", system_message),  
    ]  
    
    res = _llm.invoke(messages)
    response = res.content
    return response # Return the complete streamed output
 