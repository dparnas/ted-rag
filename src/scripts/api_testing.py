import requests

r = requests.post(
    "https://ted-rag-sandy.vercel.app/api/prompt",
    json={"question": "List three TED talks about AI"}
)

print(r.json())
