import os
from langchain_community.document_loaders import PyMuPDFLoader

base_folder = r"F:\Rag poject AI\goverment scheme project"

documents = []

replacements = {
    "PI\\/": "PM",
    "PI/": "PM",
    "PIV": "PM",
}

for root, dirs, files in os.walk(base_folder):
    for file in files:
        if file.endswith(".pdf"):
            pdf_path = os.path.join(root, file)

            loader = PyMuPDFLoader(pdf_path)
            docs = loader.load()

            # Add metadata and clean text
            scheme_name = os.path.basename(root)

            for doc in docs:
                text = doc.page_content

                # Replace incorrect text
                for wrong, correct in replacements.items():
                    text = text.replace(wrong, correct)

                doc.page_content = text

                doc.metadata["scheme"] = scheme_name
                doc.metadata["file_name"] = file

            documents.extend(docs)

# print("Total Pages:", len(documents))