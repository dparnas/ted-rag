// src/pages/api/prompt.

import type { NextApiRequest, NextApiResponse } from "next";
import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai"

type PromptRequest = { question: string };

type ContextChunk = {
  talk_id: string;
  title: string;
  chunk: string;
  score: number;
  // Internal-only (not returned in `context`)
  topics?: string;
  related_talks?: string;
};

const DEFAULT_TOP_K = 10;
const MAX_TOP_K = 30;
const PINECONE_NAMESPACE = "testing1";

const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL ?? "RPRTHPB-text-embedding-3-small";
const CHAT_MODEL = process.env.CHAT_MODEL ?? "RPRTHPB-gpt-5-mini";

const SYSTEM_PROMPT = `You are a TED Talk assistant that answers questions strictly and
only based on the TED dataset context provided to you (metadata
and transcript passages). {{TASK_INSTR}} You must not use any external
knowledge, the open internet, or information that is not explicitly
contained in the retrieved context. If the answer cannot be
determined from the provided context, respond: “I don’t know
based on the provided TED data.” Always explain your answer
using the given context, quoting or paraphrasing the relevant
transcript or metadata when helpful.
`;

const AGENT_SYSTEM_PROMPT = `You are a router for a TED Talk RAG system.
Classify the user's question into exactly one task_id in {1,2,3,4}:

1 = Precise Fact Retrieval (one specific talk/entity/fact)
2 = Multi-Result Topic Listing (must return EXACTLY 3 DISTINCT talk titles)
3 = Key Idea Summary Extraction (find a talk and summarize its key idea)
4 = Recommendation with Evidence-Based Justification (recommend ONE talk + justify)

Then rewrite the user's question into a concise retrieval query that will help semantic search.
Output STRICT JSON with keys: task_id (int), refined_query (string), rationale (string <= 20 words).
No extra keys, no markdown.
If the question is unrelated to TED talks return 5 in task_id, a response in refined_query, an explanation in rationale.
`;

function buildTaskInstr(taskId: number): string {
  switch (taskId) {
    case 1:
      return "Your goal is to locate a single, specific entity or fact based on the prompt.";
    case 2:
      return "Your goal is to return up to 3 talk titles that match a theme or topic. Do not return multiple texts of the same talk.";
    case 3:
      return "Your goal is to identify a relevant talk and generate a concise summary based on texts from that talk.";
    case 4:
      return "Your goal is to recommend ONE talk based on the prompt and justify with evidence (quotes/paraphrases) from the context.";
    default:
      return "";
  }
}

async function agentRefineQuestion(openai_key: string, question: string) {
  //await openai.responses.create({
  const agent = new ChatOpenAI({
    apiKey: openai_key,
    model: CHAT_MODEL,
    reasoning: { effort: "medium" },
    configuration: {baseURL: "https://api.llmod.ai/v1"},
  });

  const raw = await agent.invoke([{ role: "system", content: AGENT_SYSTEM_PROMPT },
      { role: "user", content: question },
    ]).content ?? "";
      // resp.output_text ?? "";
  try {
    const parsed = JSON.parse(raw);
    const task_id = Number(parsed.task_id);
    const refined_query = String(parsed.refined_query ?? "").trim();
    const rationale = String(parsed.rationale ?? "").trim();

    if (![1, 2, 3, 4, 5].includes(task_id) || !refined_query) throw new Error("bad router output");
    return { task_id, refined_query, rationale, raw };
  } catch {
    return { task_id: 5, refined_query: question, rationale: "fallback", raw };
  }
}

async function embedQuery(openai_key: string, text: string): Promise<number[]> {
  const emb_model = new OpenAIEmbeddings({
  apiKey: openai_key,
  configuration: {
    baseURL: "https://api.llmod.ai/v1"
  },
  model: EMBEDDING_MODEL,
});

  const vec = await emb_model.embed_query(text).data?.[0]?.embedding as number[] | undefined;
  if (!vec) throw new Error("Embedding failed");
  return vec;
}

//TODO: add failsafe for at least three talks
async function queryPinecone(indexName: string, vector: number[], topK: number): Promise<ContextChunk[]> {
  const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY! });
  const index = pc.index(indexName).namespace(PINECONE_NAMESPACE);

  const res = await index.query({
    vector,
    topK,
    includeMetadata: true,
  });

  const matches = res.matches ?? [];
  return matches.map((m) => {
    const md: any = m.metadata ?? {};
    return {
      talk_id: String(md.talk_id ?? ""),
      title: String(md.title ?? ""),
      topics: String(md.topics ?? ""),
      related_talks: String(md.related_talks ?? ""),
      chunk: String(md.chunk ?? ""),
      score: Number(m.score ?? 0),
    };
  });
}

function distinctTalkCount(contexts: ContextChunk[]) {
  return new Set(contexts.map((c) => c.talk_id).filter(Boolean)).size;
}

function buildAugmentedPrompt(question: string, refinedQuery: string, contexts: ContextChunk[], taskId: number) {
  const system = SYSTEM_PROMPT.replace("{{TASK_INSTR}}", buildTaskInstr(taskId));

  const ctxBlock = contexts
    .map((c, i) =>
      [
        `Context ${i + 1}:`,
        `talk_id: ${c.talk_id}`,
        `title: ${c.title}`,
        `topics: ${c.topics ?? ""}`,
        `related_talks: ${c.related_talks ?? ""}`,
        `chunk: ${c.chunk}`,
        `score: ${c.score}`,
      ].join("\n")
    )
    .join("\n\n");

  const user = [
    `Original Question: ${question}`,
    `Agent Modified Question: ${refinedQuery}`,
    `Use ONLY the context below.`,
    "",
    ctxBlock,
  ].join("\n");

  return { System: system, User: user };
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ error: "Method Not Allowed. Use POST." });
  }

  try {
    const body = req.body as Partial<PromptRequest>;
    const question = String(body?.question ?? "").trim();
    if (!question) return res.status(400).json({ error: "Missing 'question' in JSON body" });

    const OPENAI_API_KEY = process.env.LLMOD_API_KEY;
    const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
    const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME;

    if (!OPENAI_API_KEY) return res.status(500).json({ error: "Server missing OPENAI_API_KEY" });
    if (!PINECONE_API_KEY) return res.status(500).json({ error: "Server missing PINECONE_API_KEY" });
    if (!PINECONE_INDEX_NAME) return res.status(500).json({ error: "Server missing PINECONE_INDEX_NAME" });

    // const openai = new OpenAI({ apiKey: OPENAI_API_KEY, baseURL: "https://api.llmod.ai/v1",});

    // 1) Agent step: classify + refine
    const agent = await agentRefineQuestion(OPENAI_API_KEY, question);

    // If router says unrelated, still return required schema
    if (agent.task_id === 5) {
      return res.status(200).json({
        response: "I don’t know based on the provided TED data.",
        context: [],
        Augmented_prompt: {
          System: SYSTEM_PROMPT.replace("{{TASK_INSTR}}", ""),
          User: question,
        },
      });
    }

    // 2) Retrieval (task-aware)
    const initialTopK = agent.task_id === 2 ? Math.min(MAX_TOP_K, Math.max(DEFAULT_TOP_K, 15)) : DEFAULT_TOP_K;

    const qVec = await embedQuery(OPENAI_API_KEY, agent.refined_query);

    let contexts = await queryPinecone(PINECONE_INDEX_NAME, qVec, initialTopK);

    // For task 2: ensure we *can* produce 3 distinct titles by widening retrieval
    if (agent.task_id === 2) {
      let k = initialTopK;
      while (distinctTalkCount(contexts) < 3 && k < MAX_TOP_K) {
        contexts = await queryPinecone(PINECONE_INDEX_NAME, qVec, k);
        k = Math.min(MAX_TOP_K, k + 5);
      }
    }

    // 3) Build augmented prompt
    const augmented = buildAugmentedPrompt(question, agent.refined_query, contexts, agent.task_id);

    // 4) LLM answer
    // const llm = await openai.responses.create({
    //   model: CHAT_MODEL,
    //   // temperature: 0.2,
    //   input: [
    //     { role: "system", content: augmented.System },
    //     { role: "user", content: augmented.User },
    //   ],
    // });

    const llm = new ChatOpenAI({
      apiKey: OPENAI_API_KEY,
      configuration: {
        baseURL: "https://api.llmod.ai/v1"
      },
      model: CHAT_MODEL,
    });

    const responseText = llm.invoke([{ role: "system", content: augmented.System },
        { role: "user", content: augmented.User },]) ?? "I don’t know based on the provided TED data.";
    // llm.output_text ?? "I don’t know based on the provided TED data.";

    // 5) Return required JSON schema
    return res.status(200).json({
      response: responseText,
      context: contexts.map((c) => ({
        talk_id: c.talk_id,
        title: c.title,
        chunk: c.chunk,
        score: c.score,
      })),
      Augmented_prompt: augmented,
    });
  } catch (err: any) {
    return res.status(500).json({ error: `Server error: ${err?.message ?? String(err)}` });
  }
}
