# Qwen Function Calling

## 一. 简介

- 记录使用 `Qwen2.5` 的 `Function Calling` 功能

## 二. 运行
下面是 3 种运行方式：
- Hugging Face transformers
- Ollama
- vLLM

### 1. Hugging Face transformers

- 基于 `transformers` 库运行

#### 1.1 环境安装

- 在 `conda` 虚拟环境或 `docker` 容器内执行：

```bash
pip install -i https://mirrors.aliyun.com/pypi/simple transformers==4.48.2
```

#### 1.2 下载模型

- 从 `huggingface` 上下载 `Qwen2.5-7B-Instruct` 模型，放到本地目录，如： `/home/dhd/models` 

#### 1.3  运行 Function Calling 代码

```bash
python hf_transformers.py
```

记得把代码中的 `model_name_or_path` 替换为对应的模型路径

### 2. Ollama

- 使用 `Ollama` 启动 `Qwen2.5` 的服务，结合 `ollama` 的 `python` 客户端，运行  `Function Calling` 功能

#### 2.1 创建 ollama 服务

- 按照：ollama 的安装和使用.md，构建好服务及客户端环境

#### 2.2 运行 Function Calling 代码

- 在客户端环境中，执行：

```bash
python ollama_demo.py
```

### 3. vLLM

- 使用 `vllm` 启动 `Qwen2.5` 的服务，结合 `openai` 的 `python` 客户端，运行  `Function Calling` 功能

#### 3.1 创建 vllm 服务

- 按照：vllm 的安装和使用.md，构建好服务及客户端环境

#### 3.2 运行 Function Calling 代码

- 在客户端环境中，执行：

```bash
python vllm_demo.py
```

## 三. 参考


- https://qwen.readthedocs.io/en/v2.5/framework/function_call.html




