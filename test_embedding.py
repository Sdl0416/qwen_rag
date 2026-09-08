import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)

response = client.embeddings.create(
    model="qwen3.7-text-embedding",
    input="人工智能是一门重要的技术。",
    dimensions=1024
)

vector = response.data[0].embedding

print("向量维度：", len(vector))
print("前10个数：", vector[:10])
