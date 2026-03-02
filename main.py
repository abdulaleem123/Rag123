from pypdf import PdfReader

# PDF path (change filename as needed)
pdf_path = r"C:\Users\Abdul\Documents\sample.pdf"

reader = PdfReader(pdf_path)

text = ""
for page in reader.pages:
    text += page.extract_text()

print("PDF CONTENT:\n")
print(text[:2000])  # print first 2000 chars

import os
os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY"

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

pdf_path = r"C:\Users\Abdul\Documents\sample.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)
print(f"Total Chunks: {len(chunks)}")

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

from langchain_chroma import Chroma

vectorstore = Chroma(
    collection_name="pdf_rag",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

vectorstore.add_documents(chunks)
vectorstore.persist()

print("✅ Data stored in ChromaDB")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

query = "Explain the main topic discussed in the PDF"
docs = retriever.get_relevant_documents(query)

context = "\n\n".join([doc.page_content for doc in docs]

)



import google.generativeai as genai
import os

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

model = genai.GenerativeModel("gemini-1.5-flash")

prompt = """
CONTEXT:
This document explains the importance of a balanced diet. It describes how carbohydrates
provide energy, proteins help in muscle growth and repair, and fats support brain function.
The document also highlights the role of vitamins and minerals in maintaining immunity
and overall health. It warns that excessive consumption of processed food can lead to
obesity, diabetes, and heart disease.

QUESTION:
According to the document, why is a balanced diet important for human health?

INSTRUCTIONS:
- Answer strictly using the provided context
- Do not add external knowledge
- If the answer is not found, say:
  "The document does not provide this information."

ANSWER:
"""

response = model.generate_content(prompt)

print("\n🤖 RAG Answer:\n")
print(response.text)