from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "llama-3.3-70b-versatile"


def llm_call(messages,
             temperature=0,
             max_tokens=256):
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # print("=" * 80)
    # print("FINISH REASON:", response.choices[0].finish_reason)
    # print("MESSAGE:", response.choices[0].message)
    # print("CONTENT:", repr(response.choices[0].message.content))
    # print("=" * 80)


        return response.choices[0].message.content
    except Exception as e:
        if "429" in str(e):
            return "⚠️ Rate limit reached. Please try again later."
        else:
            raise e
