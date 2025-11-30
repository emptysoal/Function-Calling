import json
from openai import OpenAI

from tools import get_function_by_name, TOOLS

MESSAGES = [
    {"role": "system",
     "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30"},
    {"role": "user", "content": "What's the temperature in San Francisco now? How about tomorrow?"},
]

tools = TOOLS
messages = MESSAGES[:]

openai_api_key = "EMPTY"
openai_api_base = "http://192.168.0.230:32804/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

model_name = "/workspace/text-generation-webui-customized/models/Qwen2.5-7B-Instruct"

response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    tools=tools,
    temperature=0.7,
    top_p=0.8,
    max_tokens=512,
    extra_body={
        "repetition_penalty": 1.05,
    },
)

# print(response.choices[0])


messages.append(response.choices[0].message.model_dump())

if tool_calls := messages[-1].get("tool_calls", None):
    for tool_call in tool_calls:
        call_id: str = tool_call["id"]
        if fn_call := tool_call.get("function"):
            fn_name: str = fn_call["name"]
            fn_args: dict = json.loads(fn_call["arguments"])

            fn_res: str = json.dumps(get_function_by_name(fn_name)(**fn_args))

            messages.append({
                "role": "tool",
                "content": fn_res,
                "tool_call_id": call_id,
            })

response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    tools=tools,
    temperature=0.7,
    top_p=0.8,
    max_tokens=512,
    extra_body={
        "repetition_penalty": 1.05,
    },
)

print(response.choices[0].message.content)
