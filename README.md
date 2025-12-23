This is a [Next.js](https://nextjs.org) project facilitating a RAG Agent LLM to answer questions about a TED talk database.
The app has two https endpoints at https://ted-rag-sandy.vercel.app:
1. api/prompt - which allows the user to ask a question against the TED database and recieve an LLM created response.
2. api/stats - which returns a JSON of the architecture's basic parameters

These may be accessed via POST (api/prompt) and GET (api/stats).
Example Python code is provided:

import requests
q = "Your question"
# ask question against TED database
r = requests.post(
     "https://ted-rag-sandy.vercel.app/api/prompt",
     json={"question": q}
)
print(r.json())

# get parameters
r = requests.get("https://ted-rag-sandy.vercel.app/api/stats")

print(r.json())
