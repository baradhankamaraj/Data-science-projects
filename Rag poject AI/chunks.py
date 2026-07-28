from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion import documents

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 800,
    chunk_overlap = 150
)

chunks = text_splitter.split_documents(documents)

# print(f" length of chunks : {len(chunks)}")