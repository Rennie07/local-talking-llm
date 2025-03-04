from langchain_core.globals import set_verbose, set_debug
from langchain.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.prompts import ChatPromptTemplate

set_debug(True)
set_verbose(True)

class ChatPDF:
    def __init__(self, llm_model: str = "qwen2.5"):
        self.model = ChatOllama(model=llm_model)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        
        # Define the prompt
        self.prompt = ChatPromptTemplate.from_template(
            "Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}"
        )

        self.vector_store = None
        self.retriever = None
        self.chain = None

    def ingest(self, pdf_file_path: str):
        """Loads a PDF, splits text into chunks, and stores embeddings in a vector database."""
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=FastEmbedEmbeddings(),
            persist_directory="chroma_db"
        )

    def ask(self, query: str):
        """Retrieves relevant context from the vector store and generates an answer."""
        if not self.vector_store:
            return "Please upload a PDF document first."

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 10, "score_threshold": 0.0}
        )

        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        )

        print("Query:", query)
        print("Chain:", self.chain)

        return self.chain.invoke(query)

    def clear(self):
        """Clears stored data."""
        self.vector_store = None
        self.retriever = None
        self.chain = None
