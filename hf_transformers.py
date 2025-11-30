import re
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

from tools import get_function_by_name, TOOLS

MESSAGES = [
    {"role": "system",
     "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30"},
    {"role": "user", "content": "What's the temperature in San Francisco now? How about tomorrow?"},
]

model_name_or_path = "/workspace/text-generation-webui-customized/models/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype="auto",
    device_map="auto",
)

tools = TOOLS
messages = MESSAGES[:]

text = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
output_text = tokenizer.batch_decode(outputs)[0][len(text):]
"""
print(output_text)
The output texts should be like:

<tool_call>
{"name": "get_current_temperature", "arguments": {"location": "San Francisco, CA, USA"}}
</tool_call>
<tool_call>
{"name": "get_temperature_date", "arguments": {"location": "San Francisco, CA, USA", "date": "2024-10-01"}}
</tool_call><|im_end|>
"""


def try_parse_tool_calls(content: str):
    """Try parse the tool calls."""
    tool_calls = []
    offset = 0
    for i, m in enumerate(re.finditer(r"<tool_call>\n(.+)?\n</tool_call>", content)):
        if i == 0:
            offset = m.start()
        try:
            func = json.loads(m.group(1))
            tool_calls.append({"type": "function", "function": func})
            if isinstance(func["arguments"], str):
                func["arguments"] = json.loads(func["arguments"])
        except json.JSONDecodeError as e:
            print(f"Failed to parse tool calls: the content is {m.group(1)} and {e}")
            pass
    if tool_calls:
        if offset > 0 and content[:offset].strip():
            c = content[:offset]
        else:
            c = ""
        return {"role": "assistant", "content": c, "tool_calls": tool_calls}
    return {"role": "assistant", "content": re.sub(r"<\|im_end\|>$", "", content)}


messages.append(try_parse_tool_calls(output_text))

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

"""
print(messages)
[
    {'role': 'system', 'content': 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30'},
    {'role': 'user', 'content': "What's the temperature in San Francisco now? How about tomorrow?"},
    {'role': 'assistant', 'content': '', 'tool_calls': [
        {'type': 'function', 'function': {'name': 'get_current_temperature', 'arguments': {'location': 'San Francisco, CA, USA'}}}, 
        {'type': 'function', 'function': {'name': 'get_temperature_date', 'arguments': {'location': 'San Francisco, CA, USA', 'date': '2024-10-01'}}},
    ]},
    {'role': 'tool', 'name': 'get_current_temperature', 'content': '{"temperature": 26.1, "location": "San Francisco, CA, USA", "unit": "celsius"}'},
    {'role': 'tool', 'name': 'get_temperature_date', 'content': '{"temperature": 25.9, "location": "San Francisco, CA, USA", "date": "2024-10-01", "unit": "celsius"}'},
]
"""

text = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
output_text = tokenizer.batch_decode(outputs)[0][len(text):]
print(output_text)
