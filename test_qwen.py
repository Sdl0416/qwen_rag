import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)

response = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {
            "role": "user",
            "content": "你好，请用一句话介绍人工智能。"
        }
    ]
)

print(response.choices[0].message.content)
