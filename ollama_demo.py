import json
import ollama

from tools import get_function_by_name, TOOLS

MESSAGES = [
    {"role": "system",
     "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30"},
    {"role": "user", "content": "What's the temperature in San Francisco now? How about tomorrow?"},
]

tools = TOOLS
messages = MESSAGES[:]

model_name = "qwen2.5-7b-local"

# 创建客户端并指定远程 Ollama 服务的地址
client = ollama.Client(host='http://192.168.0.230:32803')

# 使用 chat 方法与模型对话
response = client.chat(
    model=model_name,
    messages=messages,
    tools=tools
)
# print(response)


messages.append(response["message"])

if tool_calls := messages[-1].get("tool_calls", None):
    for tool_call in tool_calls:
        if fn_call := tool_call.get("function"):
            fn_name: str = fn_call["name"]
            fn_args: dict = fn_call["arguments"]

            fn_res: str = json.dumps(get_function_by_name(fn_name)(**fn_args))

            messages.append({
                "role": "tool",
                "name": fn_name,
                "content": fn_res,
            })

response = client.chat(
    model=model_name,
    messages=messages,
    tools=tools,
)
print(response["message"])
