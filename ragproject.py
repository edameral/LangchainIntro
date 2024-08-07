import bs4
from dotenv import load_dotenv
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model_= "gpt-3.5 turbo")

# Load the specified web page and retrieve the necessary information.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    )
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_text(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings)

retriever = vectorstore.as_retriever()
#rag prompt
prompt= hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context":retriever | format_docs, "question":RunnablePassthrough()}
    | prompt
    | llm
    |StrOutputParser
)


if __name__ == "__main__":
    for chunk in rag_chain.stream("What is maximum inner product search?"):
        print(chunk, end="", flush=True)
