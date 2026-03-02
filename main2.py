pip install pandas chromadb google-generativeai langchain langchain-chroma

import pandas as pd

csv_path = r"C:\data\sample.csv"
df = pd.read_csv(csv_path)

print(df)


from langchain.schema import Document

documents = []

for _, row in df.iterrows():
    text = ", ".join([f"{col}: {row[col]}" for col in df.columns])
    documents.append(Document(page_content=text))

print(documents[0])


import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings

os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY"

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)
from langchain_chroma import Chroma

vectorstore = Chroma(
    collection_name="csv_rag",
    embedding_function=embeddings,
    persist_directory="./chroma_csv_db"
)

vectorstore.add_documents(documents)
vectorstore.persist()

print("CSV data stored in ChromaDB")

query = "Who is the oldest person in the CSV file?"

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
docs = retriever.get_relevant_documents(query)

context = "\n".join(doc.page_content for doc in docs)

print("Retrieved Context:\n", context)

import google.generativeai as genai

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

model = genai.GenerativeModel("gemini-1.5-flash")

prompt = f"""
CONTEXT:
{context}

QUESTION:
{query}

INSTRUCTIONS:
- Answer only using the context
- If answer is not present, say:
  "The CSV does not contain this information."

ANSWER:
"""

response = model.generate_content(prompt)

print("\n🤖 RAG Answer:\n")
print(response.text)

