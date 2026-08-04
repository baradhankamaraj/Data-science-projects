from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from embeddings import embedds
from chunks import chunks

vectordb = FAISS.from_documents(
    documents=chunks,
    embedding=embedds,
    distance_strategy=DistanceStrategy.COSINE,
    normalize_L2=True
)

vectordb.save_local("vectorstore")

vectordb = FAISS.load_local(
    "vectorstore",
    embedds,
    allow_dangerous_deserialization=True
)
# docs = retriever.invoke("what is GST?")