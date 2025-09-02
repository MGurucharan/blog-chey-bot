from langchain_community.document_loaders import MongodbLoader
from bs4 import BeautifulSoup
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
import asyncio
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from pydantic import BaseModel
load_dotenv()
from fastapi import FastAPI
from openai import OpenAI
import os
import asyncio
import re
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_title_and_content(html_content):
    """
    Extracts the title (first text before first <p> tag) and cleans the rest.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract title: get first text node before <p>
    title = None
    if soup.p:
        # Grab all text before the first <p>
        possible_title = soup.get_text(separator="\n", strip=True).split("\n")[0]
        title = possible_title.strip()

    # Remove scripts/styles/images
    for tag in soup(['script', 'style', 'img']):
        tag.decompose()

    plain_text = soup.get_text(separator=' ', strip=True)

    return title, plain_text


# Global Pinecone + Embedding initialization (done once on server startup)

api_key = os.getenv("OPENAI_API_KEY")
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)

pinecone_api_key = os.environ["PINECONE_API_KEY"]
pc = Pinecone(api_key=pinecone_api_key)
index_name = "blogguru-index"
dimension = 3072

# Ensure index exists (only create if not present)
if index_name not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = PineconeVectorStore(
    index_name=index_name,
    embedding=embedding_model
)


@app.get("/")
def func():
    return {"FastAPI Chatbot running !"}

@app.post("/fastapi/ingest")  
async def ingester():
        loader=MongodbLoader(
        connection_string=os.getenv("MONGO_URI"),
        db_name="blogDB",
        collection_name="items",
        field_names=["title","content"]
    )

        documents= await loader.aload()

    # Extract all the plain_text and title from the documents of MongoDB 
    # Convert each document's plain text into text chunks 
    # For a given document , for given chunks , attach a title as metadata to filter out later
    # Store the document chunks ( ones with title as metadata) and text chunks in a separate file for reference 
    
        text_splitter = SemanticChunker(embedding_model)
        separated_doc_chunks=[]
        separated_text_chunks=[]

        for doc in documents:
            extracted_title,plain_text = extract_title_and_content(doc.page_content)
            chunks=text_splitter.create_documents([plain_text])
            for chunk in chunks:
                chunk.metadata["title"]=extracted_title
                separated_doc_chunks.append(chunk)
            for ch in chunks:
                separated_text_chunks.append(ch.page_content)

    
        articles="\n\n---------------------------------------------------------------------------------------\n\n".join(separated_text_chunks)
    
    

        # 6. Insert documents using LangChain wrapper

        PineconeVectorStore.from_documents(
            separated_doc_chunks,  # Your LangChain Document[] list
            embedding=embedding_model,
            index_name=index_name
        )
        

        return {"✅ Pinecone index created and populated with 3072-d embeddings."}
        
        # await asyncio.sleep(5)  # ✅ non-blocking wait
        
        # stats = pc.Index(index_name).describe_index_stats()
        # print("Vector count:", stats.total_vector_count)
        
        # Chain creation and finalization

class Query(BaseModel):
    title:str
    question:str

@app.post("/fastapi/post")
async def poster(query:Query) :
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        chain = load_qa_chain(llm, chain_type='stuff')
            
        template = """
        You are an expert on answering questions related to blog posts.
        Use the following context to answer the question given by the user.

        If the user understands and replies back , be polite and give a soothing response to the user.  
        If the context includes relevant blog content, summarize it.
        If nothing is found, just say: "Could you be more specific?"

        Context:
        {context}

        Question: {question}
        Answer:
        """

        prompt=PromptTemplate(
            template=template,
            input_variables=["context","question"]
        )

        def retriever_with_title(question, title=None):
            if title:
                # Better to use metadata filter instead of injecting title into query string
                return index.as_retriever(
                    search_kwargs={"filter": {"title": title}}
                ).invoke(question)
            return index.as_retriever().invoke(question)

        docs=retriever_with_title(query.question,query.title)
        print("Retrieved docs : ",docs)
        
        rag_chain = (
            {
                "context": lambda q: retriever_with_title(q, title=query.title),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        result = rag_chain.invoke(query.question)
        return {"message":result}

