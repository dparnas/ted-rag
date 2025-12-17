import os
import json
from typing import Any, Dict, List
import joblib
api_keys = joblib.load('../data/api_keys.pkl')

from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

TOP_K = 5
llmod_key = api_keys['LLMOD_KEY']
llmod_base_url = "https://api.llmod.ai"
llmod_embedding_model = "RPRTHPB-text-embedding-3-small"
llmod_chat_model = "RPRTHPB-gpt-5-mini"

PINECONE_API_KEY = api_keys['PINECONE_API_KEY']
PINECONE_INDEX_NAME = "ted-rag"

SYSTEM_PROMPT = """You are a TED Talk assistant that answers questions strictly and
only based on the TED dataset context provided to you (metadata
and transcript passages). You must not use any external
knowledge, the open internet, or information that is not explicitly
contained in the retrieved context. If the answer cannot be
determined from the provided context, respond: “I don’t know
based on the provided TED data.” Always explain your answer
using the given context, quoting or paraphrasing the relevant
transcript or metadata when helpful.
"""

#todo: remove at deployment
question_embedding_dict = joblib.load('../data/question_embeddings.pkl')
question_context_dict = joblib.load('../data/question_contexts.pkl')
# ----------------------------
# LangChain clients
# ----------------------------


# ----------------------------
# Pinecone client
# ----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
print("Initialized Pinecone index")

def retrieve_contexts(question: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """Embed the question, query Pinecone, and return normalized context objects."""
    #todo: remove at deployment
    print("embedding question")
    previous_embedding = question_embedding_dict.get(question, None)
    if previous_embedding is None:
        #todo: move when more than one question
        embeddings = OpenAIEmbeddings(
            model=llmod_embedding_model,  # Your Azure deployment name
            base_url=llmod_base_url,
            api_key=llmod_key
        )
        print("initialized embedding model")

        q_vec = embeddings.embed_query(question)  # list[float]
        question_embedding_dict[question] = q_vec
        joblib.dump(question_embedding_dict, '../data/question_embeddings.pkl')

    query_kwargs = dict(
        vector=q_vec,
        top_k=top_k,
        include_metadata=True,
    )

    print("querying Vector DB")
    previous_res = question_context_dict.get(question, None)
    if previous_res is None:
        res = index.query(**query_kwargs)
        question_context_dict[question] = res
        joblib.dump(question_context_dict, '../data/question_contexts.pkl')

    contexts = []
    for m in (res.matches or []):
        md = m.metadata or {}
        contexts.append({
            "talk_id": str(md.get("talk_id", "")),
            "title": str(md.get("title", "")),
            "topics": str(md.get("topics", "")),
            "related_talks": str(md.get("related_talks", "")),
            "chunk": str(md.get("chunk", "")),
            "score": float(m.score or 0.0),
        })
    return contexts


def build_augmented_prompt(question: str, contexts: List[Dict[str, Any]]) -> Dict[str, str]:
    print("Building augmented prompt")
    system = SYSTEM_PROMPT

    ctx_block = "\n\n".join(
        f"Context {i+1}:\n"
        f"title: {c['title']}\n"
        f"topics: {c['topics']}\n"
        f"related_talks: {c['related_talks']}"
        f"chunk: {c['chunk']}"
        for i, c in enumerate(contexts)
    )

    user = f"Question: {question}\n\nUse ONLY the context below.\n\n{ctx_block}"
    return {"System": system, "User": user}


def ask_rag(question: str, stop_before_llm=False) -> Dict[str, Any]:
    contexts = retrieve_contexts(question, top_k=TOP_K)
    aug = build_augmented_prompt(question, contexts)

    if stop_before_llm:
        return {
        "question": question,
        "response": None,
        "context": contexts,
        "Augmented_prompt": aug,
        }

    messages = [
        SystemMessage(content=aug["System"]),
        HumanMessage(content=aug["User"]),
    ]
    #todo: move when testing llm
    llm = ChatOpenAI(
        model=llmod_chat_model,  # Your Azure deployment name
        base_url=llmod_base_url,
        api_key=llmod_key
    )
    print("initialized llm model")

    print("Invoking llm")

    answer = llm.invoke(messages).content

    return {
        "question": question,
        "response": answer,
        "context": contexts,
        "Augmented_prompt": aug,
    }

if __name__ == "__main__":
    q = input("I am looking for a TED talk about climate change and what individuals can do in their daily lives. Which talk would you recommend?").strip()
    out = ask_rag(q, stop_before_llm=True)

    print("\n=== Retrieved contexts (top_k) ===")
    for i, c in enumerate(out["context"], start=1):
        print(f"\n[{i}] score={c['score']:.4f} | talk_id={c['talk_id']} | title={c['title']}")
        print(c["chunk"][:500] + ("..." if len(c["chunk"]) > 500 else ""))

    print("\n=== Augmented Prompt ===")
    print("\n[SYSTEM]\n" + out["Augmented_prompt"]["System"])
    print("\n[USER]\n" + out["Augmented_prompt"]["User"][:2000] + ("..." if len(out["Augmented_prompt"]["User"]) > 2000 else ""))

    print("\n=== LLM Response ===")
    print(out["response"])

