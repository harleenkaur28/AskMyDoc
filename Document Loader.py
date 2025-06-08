from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
import streamlit as st
import tempfile
import uuid

embedding_model= HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2' )
llm= ChatGoogleGenerativeAI(model="gemini-2.0-flash")
parser= StrOutputParser()

prompt=PromptTemplate(
  template="You are a helpful assistant. From the provided context, answer the question{question} context: {context} The previous chat history is also provided {message_history}",
  input_variables= ['context', 'question', 'message_history']
)
def upload_file(temp_path):
        loader= PyPDFLoader(temp_path)
        docs=loader.load()
        return docs

def create_chunks(docs):
  splitter= RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
  chunks= splitter.split_documents(docs)

  return chunks

def embedding_generation(chunks):
  vector_store = Chroma.from_documents(
      documents=chunks,
      embedding=embedding_model,
      collection_name=str(uuid.uuid4())  
  )
  return vector_store

def format_docs(retrieved_docs):
  return "\n".join(doc.page_content for doc in retrieved_docs)

st.header("Document Q/A Chatbot")
uploaded_file= st.file_uploader(label="File upload")
if uploaded_file is not None:
    if ("current_file" not in st.session_state) or (uploaded_file.name != st.session_state.current_file):
        st.session_state.current_file = uploaded_file.name
        st.session_state.message_history = []  
   
    st.write(f"Current file: {st.session_state.current_file}")  

    if "message_history" not in st.session_state:
        st.session_state.message_history = []

    if uploaded_file.type in ['application/pdf', 'text/plain']:
        suffix = ".pdf" if uploaded_file.type == 'application/pdf' else ".txt"
        with tempfile.NamedTemporaryFile(delete=True, suffix= suffix) as file:
            uploaded_file.seek(0)
            file.write(uploaded_file.getbuffer())
            temp_path = file.name

            st.subheader("File loaded")

            question= st.text_input("Ask a Question")

            initial_chain = RunnableParallel({
                "question": RunnableLambda(lambda x: x["question"]),
                "vs": RunnableLambda(lambda x: x["uploaded_file"]) 
                    | RunnableLambda(create_chunks) 
                    | RunnableLambda(embedding_generation)
            })

            context_chain = initial_chain | RunnableLambda(lambda d: d["vs"].as_retriever(search_type="mmr", search_kwargs={"k": 4}).get_relevant_documents(d["question"])) | RunnableLambda(format_docs)

            parallel_chain = RunnableParallel({
                "question": RunnableLambda(lambda x: x["question"]),
                "context": context_chain,
                "message_history": RunnableLambda(lambda x: x["message_history"])
            })

            final_chain = parallel_chain | prompt | llm | parser
            
            if st.button("ASK"):
               docs=upload_file(temp_path)

               result=final_chain.invoke({'question':question, 'message_history': st.session_state.message_history, 'uploaded_file':docs})

               st.session_state.message_history.append(HumanMessage(content=question))
               st.session_state.message_history.append(AIMessage(content=result))
               st.subheader(result)
            #  st.text(st.session_state.message_history)
               file.close()
               