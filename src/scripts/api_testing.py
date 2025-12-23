import requests
q1 = "I am looking for a TED talk about climate change and what individuals can do in their daily lives. Which talk would you recommend?"
q2 = "Find a TED talk where the speaker talks about creativity. Provide the title, speaker and a short summary of the speech."
q3 = "Which TED talks focus on education or learning? Return a list of exactly 3 talk titles."
q4 = "Find a TED talk about Rick and Morty and provide a short description of the talk." # Can't find
q5 = "I want to learn about overcoming fear and anxiety. Help me to find a relevant TED talk and explain why its relevant."

r = requests.post(
    "https://ted-rag-sandy.vercel.app/api/prompt",
    json={"question": q2}
)

# r = requests.get("https://ted-rag-sandy.vercel.app/api/stats")
response_data = r.json()
print("RESPONSE")
print(response_data['response'])
print('---------------------------------')
print(f"CONTEXT - {len(response_data['context'])}")
print(response_data['context'][0])
print(response_data['context'][-1])
print('---------------------------------')
print("AUGMENTED_PROMPT")
# print(response_data['Augmented_prompt'])
print(response_data['Augmented_prompt']['System'][:200])
print(response_data['Augmented_prompt']['User'])
print('---------------------------------')

