import os
import faiss
import pickle
import json

from pathlib import Path

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

os.environ["OPENAI_API_KEY"] = ""

ps = list(Path("data").glob("*.json"))

data = []
sources = []
for p in ps:
    with open(p, "rb") as f:
        txt_dict = json.load(f)
        for k in txt_dict.keys():
            data.append(txt_dict[k])
            sources.append(k)


text_splitter = CharacterTextSplitter(chunk_size=1000, separator=".")
docs = []
metadatas = []
for i, d in enumerate(data):
    splits = text_splitter.split_text(d)
    docs.extend(splits)
    metadatas.extend([{"source": sources[i]}] * len(splits))

bad_docs = [ i for i, d in enumerate(docs) if len(d) > 2000 ]

for i in sorted(bad_docs, reverse=True):
    print('deleting doc due to size', f'size:{len(docs[i])} doc: {docs[i]}' )
    del docs[i]
    del metadatas[i]


# Here we create a vector store from the documents and save it to disk.
store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
faiss.write_index(store.index, "data/docs.index")
store.index = None
with open("data/aifred_docs.pkl", "wb") as f:
    pickle.dump(store, f)