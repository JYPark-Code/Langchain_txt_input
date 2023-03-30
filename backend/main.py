import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
# from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI
# import magic
# import nltk


app = FastAPI()

# Add CORS support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()  # load variables from .env file
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

@app.post("/add_txts/")
async def add_txts(file: UploadFile = File(...)):
    try:
        with open(file.filename, "wb") as f:
            f.write(file.file.read())
        loader = UnstructuredFileLoader(file.filename)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        global docsearch
        docsearch = Chroma.from_documents(texts, embeddings)
        return {"message": "Text file added successfully!"}
    except Exception as e:
        print(str(e))
        return {"message": "Error adding text file"}


@app.post("/query/")
async def query(query: str):
    try:
        qa = VectorDBQA.from_chain_type(llm=ChatOpenAI(), chain_type="stuff", vectorstore=docsearch)
        result = qa.run(query)
        while result.startswith('\n'):
            result = result[1:]
        return {"answer": str(result)}

    except Exception as e:
        print(str(e))
        return {"message": "Error with query"}
