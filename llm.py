from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

##1. LLM Model with HuggingFace
llm = VertexAI(model_name="gemini-pro")

###2. Embedding Model with HuggingFace
embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@001")


###3. Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def upload_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    # cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    with open(file_path, "wb") as f:
        f.write(file_content)
    return file_path


def embed_file(file_path):
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    vectorstore = Chroma.from_documents(docs, embeddings)

    # retriever = vectorstore.as_retriever()
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 1}
    )
    return retriever


def execute_chain(retriever, user_message):
    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )
    response = chain.invoke(user_message)
    return response
