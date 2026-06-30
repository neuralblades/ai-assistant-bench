import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("HF_API_TOKEN"),
    base_url="https://router.huggingface.co/v1",
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
    max_tokens=50,
)

print(response.choices[0].message.content)