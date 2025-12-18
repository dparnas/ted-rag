import os
import json
from typing import Any, Dict, List
import joblib
api_keys = joblib.load('../data/api_keys.pkl')

from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

DEFAULT_TOP_K = 10
LLMOD_KEY = api_keys['open_ai_key'] # api_keys['LLMOD_KEY']
LLMOD_BASE_URL = "https://api.openai.com/v1" # "https://api.llmod.ai"
LLMOD_EMBEDDING_MODEL = "text-embedding-3-small" # "RPRTHPB-text-embedding-3-small"
LLMOD_CHAT_MODEL = "gpt-5-mini" # "RPRTHPB-gpt-5-mini"

PINECONE_API_KEY = api_keys['PINECONE_API_KEY']
PINECONE_INDEX_NAME = "ted-rag"

SYSTEM_PROMPT = """You are a TED Talk assistant that answers questions strictly and
only based on the TED dataset context provided to you (metadata
and transcript passages). {} You must not use any external
knowledge, the open internet, or information that is not explicitly
contained in the retrieved context. If the answer cannot be
determined from the provided context, respond: “I don’t know
based on the provided TED data.” Always explain your answer
using the given context, quoting or paraphrasing the relevant
transcript or metadata when helpful.
"""

# ----------------------------
# Agentic step (task routing + query refinement)
# ----------------------------

TASKS = {
    1: "Precise Fact Retrieval",
    2: "Multi-Result Topic Listing (exactly 3 titles)",
    3: "Key Idea Summary Extraction",
    4: "Recommendation with Evidence-Based Justification",
}

AGENT_SYSTEM_PROMPT = """You are a router for a TED Talk RAG system.
Classify the user's question into exactly one task_id in {1,2,3,4}:

1 = Precise Fact Retrieval (one specific talk/entity/fact)
2 = Multi-Result Topic Listing (must return DISTINCT talk titles)
3 = Key Idea Summary Extraction (find a talk and summarize its key idea)
4 = Recommendation with Evidence-Based Justification (recommend ONE talk + justify)

Then rewrite the user's question into a concise retrieval query that will help semantic search.
Output STRICT JSON with keys: task_id (int), refined_query (string), rationale (string <= 20 words).
No extra keys, no markdown.
If the question is unrelated to TED talks return 5 in task_id, a response in refined_query, an explanation in rationale.
"""

# ----------------------------
# LangChain clients
# ----------------------------


# ----------------------------
# Pinecone client
# ----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
print("Initialized Pinecone index")

def agent_refine_question(question: str) -> Dict[str, Any]:
    """
    Agentic step between question intake and retrieval:
    - classify question into one of the 4 functional tasks
    - produce a refined query for embedding + retrieval
    """
    llm_router = ChatOpenAI(
        model=LLMOD_CHAT_MODEL,
        base_url=LLMOD_BASE_URL,
        api_key=LLMOD_KEY,
        temperature=0,  # stable routing
        reasoning_effort="medium"
    )

    messages = [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]

    raw = llm_router.invoke(messages).content

    try:
        out = json.loads(raw)
        task_id = int(out["task_id"])
        refined = str(out["refined_query"]).strip()
        rationale = str(out.get("rationale", "")).strip()
        if task_id not in TASKS or not refined:
            raise ValueError("Invalid RAG Agent output")
        return {"task_id": task_id, "refined_query": refined, "rationale": rationale, "raw": raw}
    except Exception:
        # Fallback: safe defaults if router returns invalid JSON
        return {"task_id": 5, "refined_query": question, "rationale": "fallback", "raw": raw}


def retrieve_contexts(question: str, top_k: int = DEFAULT_TOP_K, task_id: int = None) -> List[Dict[str, Any]]:
    """Embed the question, query Pinecone, and return normalized context objects."""
    print("embedding question")
    #todo: move when more than one question
    embeddings = OpenAIEmbeddings(
        model=LLMOD_EMBEDDING_MODEL,  # Your Azure deployment name
        base_url=LLMOD_BASE_URL,
        api_key=LLMOD_KEY
    )
    print("initialized embedding model")

    q_vec = embeddings.embed_query(question)  # list[float]

    if task_id != 2:
        contexts = _get_context(q_vec, top_k)
    elif task_id == 2:
        n_talks = 0
        iterating_top_k = top_k
        while n_talks < 3 and iterating_top_k < 30:
            print("Iterating for more talks: {} talks {} chunks".format(n_talks, iterating_top_k))
            contexts = _get_context(q_vec, iterating_top_k)
            n_talks = check_number_of_talks(contexts)
            iterating_top_k += 5

    return contexts

def _get_context(q_vec, top_k):
    query_kwargs = dict(
        vector=q_vec,
        top_k=top_k,
        include_metadata=True,
        namespace="testing1"
    )

    print("querying Vector DB")
    res = index.query(**query_kwargs)
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

def check_number_of_talks(contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    talks = set([c['talk_id'] for c in contexts])
    n_talks = len(talks)
    return n_talks

def build_augmented_prompt(question: str, modified_question: str, contexts: List[Dict[str, Any]], agent: Dict[str, Any]) -> Dict[str, str]:
    print("Building augmented prompt")
    task_id = agent["task_id"]
    # add appendage to system prompt based on the 4 specific tasks
    # task-specific instructions (kept short but controlling)
    if task_id == 1:
        task_instr = "Your goal is to locate a single, specific entity or fact based on the prompt."
    elif task_id == 2:
        task_instr = "Your goal is to return up to 3 talk titles that match a theme or topic. Do not return multiple texts of the same talk."
    elif task_id == 3:
        task_instr = "Your goal is to identify a relevant talk and generate a concise summary based on texts from that talk."
    elif task_id == 4:
        task_instr = "Your goal is to recommend ONE talk based on the prompt and justify with evidence (quotes/paraphrases) from the context."
    elif task_id == 5:
        raise (f"The RAG agent identified this as not one of the 4 query capacities."
               f"Output: {agent['refined_query']}")
    modified_system_prompt = SYSTEM_PROMPT.format(task_instr)

    ctx_block = "\n\n".join(
        f"Context {i+1}:\n"
        f"title: {c['title']}\n"
        f"topics: {c['topics']}\n"
        f"related_talks: {c['related_talks']}"
        f"chunk: {c['chunk']}"
        for i, c in enumerate(contexts)
    )

    user = (f"Original Question: {question}\n"
            f"Agent Modified Question: {modified_question}\n"
            f"Use ONLY the context below.\n\n{ctx_block}")
    return {"System": modified_system_prompt, "User": user}


def ask_rag(question: str) -> Dict[str, Any]:
    agent = agent_refine_question(question)

    if agent["task_id"] == 2:
        top_k = min(30, max(DEFAULT_TOP_K, 15))
    else:
        top_k = DEFAULT_TOP_K

    # todo: handle task_id 2 inside of retrieved contexts
    contexts = retrieve_contexts(agent['refined_query'], top_k=top_k, task_id=agent['task_id'])
    aug = build_augmented_prompt(question, agent['refined_query'], contexts, agent)

    messages = [
        SystemMessage(content=aug["System"]),
        HumanMessage(content=aug["User"]),
    ]
    #todo: move when testing llm
    llm = ChatOpenAI(
        model=LLMOD_CHAT_MODEL,  # Your Azure deployment name
        base_url=LLMOD_BASE_URL,
        api_key=LLMOD_KEY
    )
    print("initialized llm model")

    print("Invoking llm")

    answer = llm.invoke(messages).content

    return {
        "question": question,
        "response": answer,
        "context": contexts,
        "Augmented_prompt": aug,
        "agent": {
            "task_id": agent["task_id"],
            "task_name": TASKS.get(agent["task_id"]),
            "refined_query": agent["refined_query"],
            "rationale": agent.get("rationale", ""),
        },
    }

if __name__ == "__main__":
    # q1 = "I am looking for a TED talk about climate change and what individuals can do in their daily lives. Which talk would you recommend?"
    # q2 = "Find a TED talk where the speaker talks about creativity. Provide the title and a short summary of the speech."
    # q3 = "Which TED talk focuses on education or learning? Return a list of exactly 3 talk titles."
    # q4 = "Find a TED talk about Rick and Morty and provide a short description of the talk." # Can't find
    q5 = "I want to learn about overcoming fear and anxiety. Help me to find a relevant TED talk and explain why its relevant."
    out = ask_rag(q5)

    print("\n=== Retrieved contexts (top_k) ===")
    for i, c in enumerate(out["context"], start=1):
        print(f"\n[{i}] score={c['score']:.4f} | talk_id={c['talk_id']} | title={c['title']}")
        print(c["chunk"][:500] + ("..." if len(c["chunk"]) > 500 else ""))

    print("\n=== Augmented Prompt ===")
    print("\n[SYSTEM]\n" + out["Augmented_prompt"]["System"])
    print("\n[USER]\n" + out["Augmented_prompt"]["User"][:2000] + ("..." if len(out["Augmented_prompt"]["User"]) > 2000 else ""))

    print("\n=== LLM Response ===")
    print(out["response"])

