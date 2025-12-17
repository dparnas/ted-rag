# querying meta-data like to get the same speech https://www.youtube.com/watch?v=ztXrf88sX-M&t=180s

import os
print(os.getcwd())
import pandas as pd
from tqdm import tqdm
import ast
import joblib
api_keys = joblib.load('../data/api_keys.pkl')

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone

llmod_key = api_keys['LLMOD_KEY']
llmod_base_url = "https://api.llmod.ai"
llmod_embedding_model = "RPRTHPB-text-embedding-3-small"
llmod_chat_model = "RPRTHPB-gpt-5-mini"

PINECONE_API_KEY = api_keys['PINECONE_API_KEY']
PINECONE_INDEX_NAME = "ted-rag"

def extract_titles(text):
    data = ast.literal_eval(text)
    titles = "; ".join(data.values())

    # Remove remaining escape characters
    return titles

embeddings = OpenAIEmbeddings(
    model=llmod_embedding_model,  # Your Azure deployment name
    base_url=llmod_base_url,
    api_key=llmod_key
)
# # Use as normal
# vector = embeddings.embed_query("Hello world")

llm = ChatOpenAI(
    model=llmod_chat_model,  # Your Azure deployment name
    base_url=llmod_base_url,
    api_key=llmod_key
)
#
# response = llm.invoke("2+2 =")
# print(response)

# ----------------------------
# Config
# ----------------------------
CSV_PATH = '../data/ted_talks_en.csv'
CHUNK_SIZE = 512 # trying https://arxiv.org/pdf/2407.01219
CHUNK_OVERLAP = 20

# ----------------------------
# Pinecone
# ----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
# print(pc.list_indexes())
index = pc.Index(PINECONE_INDEX_NAME)

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(CSV_PATH)

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

vectors = []

largest_n_chunks = 0
for j, row in tqdm(df.iterrows(), total=len(df)):
    # print(j)
    transcript = str(row["transcript"])
    if not transcript or transcript == "nan":
        continue

    chunks = splitter.split_text(transcript)
    largest_n_chunks = max(largest_n_chunks, len(chunks))
    for i, chunk in enumerate(chunks):
        vector = embeddings.embed_query(chunk)

        vectors.append({
            "id": f"{row['talk_id']}_chunk{i}",
            "values": vector,
            "metadata": {
                "talk_id": str(row["talk_id"]),
                "title": row["title"],
                "topics": row["topics"],
                "related_talks": extract_titles(row["related_talks"]),
                "chunk": chunk,
            }
        })

    # batch upserts (important)
    if len(vectors) >= 100:
        index.upsert(vectors=vectors, namespace='testing1')
        vectors = []

    if j > 10:
        break

# final flush
if vectors:
    index.upsert(vectors=vectors, namespace='testing1')
print("largest number of chunks per transcript", largest_n_chunks)
print("✅ Ingestion complete")


