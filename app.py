import streamlit as st
from utils import extract_text_from_pdf, split_text, create_vector_store
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI

st.set_page_config(page_title="PDF Chatbot")

st.title("🤖 Bot lector de PDFs")

pdfs = st.file_uploader("Subí uno o más PDFs", type="pdf", accept_multiple_files=True)
query = st.text_input("¿Qué querés preguntarle al documento?")

if pdfs and query:
    with st.spinner("Procesando..."):
        all_text = ""
        for pdf in pdfs:
            all_text += extract_text_from_pdf(pdf)
        chunks = split_text(all_text)
        db = create_vector_store(chunks, st.secrets["OPENAI_API_KEY"])
        docs = db.similarity_search(query)

        llm = OpenAI(temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)

        st.success(response)
