from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

documents = [
    "LangChain is a framework for building LLM applications.",
    "RAG stands for Retrieval Augmented Generation.",
    "ChromaDB is a vector database used to store embeddings.",
    "OpenAI provides powerful embedding and chat models."
]

splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
docs = splitter.create_documents(documents)

embedding = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(documents=docs, embedding=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

llm = ChatOpenAI(model="gpt-4o-mini")

def agentic_rag(query):
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer based on context only:
    {context}

    Question: {query}
    """

    return llm.invoke(prompt).content

if __name__ == "__main__":
    while True:
        q = input("Ask: ")
        print(agentic_rag(q))
