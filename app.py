import streamlit as st
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

# Sidebar contents
with st.sidebar:
    st.title('VUA for VEGAS Pro: PDF Query Assistant')
    st.markdown('''
    ## About
    This app is a proof of concept for MAGIX AI(LLM) integration towards market dominance strategy using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    - By: [Samuel Nyarko](samuelbj123@gmail.com)

    ''')
    add_vertical_space(5)
    st.write('VUA for VEGAS Pro: Ask Anything About Your PDF! 💡')

# Load OpenAI API key from the environment variable or use the hardcoded value if not found
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

# Store history of interactions in a session variable
if 'interaction_history' not in st.session_state:
    st.session_state.interaction_history = []

def display_chat():
    for message, speaker in st.session_state.interaction_history:
        if speaker == "VEGAS Pro User":
            st.markdown(f"<div style='background-color: #E8E8E8; padding: 10px; border-radius: 5px;'><strong>{speaker}</strong>: {message}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color: #D1F0FF; padding: 10px; border-radius: 5px;'><strong>{speaker}</strong>: {message}</div>", unsafe_allow_html=True)

def main():
    st.header("VUA for VEGAS Pro: PDF Query Assistant")

    # Display the chat history
    display_chat()

    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)

        # embeddings
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            # Set the OpenAI API key in the environment
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
        if st.button("Submit"):
            # Add user query to the chat
            st.session_state.interaction_history.append((query, "VEGAS Pro User"))
            
            with st.empty():
                st.write("Thinking...")
                
                docs = VectorStore.similarity_search(query=query, k=3)

                llm = OpenAI()
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=query)

                # Add the response to the chat
                st.session_state.interaction_history.append((response, "VEGAS Pro AI"))

                # Refresh the page to see the updated chat
                st.experimental_rerun()

if __name__ == '__main__':
    main()
