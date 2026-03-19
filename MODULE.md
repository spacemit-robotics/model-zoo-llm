# 大语言模型（LLM）

基于 OpenAI 兼容 API 的 LLM 推理组件，支持单轮/多轮对话、Tool Calling、流式输出。

## 前置条件

1. 安装推理工具：
```bash
sudo apt install llama.cpp-tools-spacemit
```

2. 下载模型（GGUF 格式）：
```bash
mkdir -p ~/.cache/models/llm
wget -P ~/.cache/models/llm https://archive.spacemit.com/spacemit-ai/model_zoo/llm/qwen2.5-0.5b-instruct-q4_0.gguf
```

可用模型列表：
| 模型 | 文件名 | 大小 |
|------|--------|------|
| Qwen2.5-0.5B | qwen2.5-0.5b-instruct-q4_0.gguf | ~400MB |
| Qwen2.5-1.5B | qwen2.5-1.5b-instruct-q4_0.gguf | ~1GB |
| Qwen2.5-3B | qwen2.5-3b-instruct-q4_0.gguf | ~2GB |
| GLM-Edge-1.5B | glm-edge-1.5b-chat-q4_0.gguf | ~1GB |
| Qwen3-30B-A3B | Qwen3-30B-A3B-Q4_0.gguf | ~17GB |

下载源：https://archive.spacemit.com/spacemit-ai/model_zoo/llm/

3. 编译 SDK 组件（llm_chat 示例）：
```bash
BUILD_TARGET_FILE=target/kx-generic-omni_agent.json ./build/build.sh package components/model_zoo/llm
```

## llm.server — 启动推理服务

启动 OpenAI 兼容的 HTTP 推理服务，供 llm.chat 等能力调用。

```bash
llama-server -m ~/.cache/models/llm/qwen2.5-0.5b-instruct-q4_0.gguf -t 8 --port 8080 &
```

参数：
- `-m`：模型文件路径（必填）
- `-t`：推理线程数，必须指定（默认 4，推荐 8，不能超过 8）
- `--port`：服务端口（默认 8080）

## llm.chat — 对话测试

通过 SDK 示例程序与大模型对话。需先启动 llm.server。

```bash
llm_chat "你好，请介绍一下自己" "http://localhost:8080/v1" "qwen2.5-0.5b" "You are a helpful assistant." 256
```

参数（按顺序）：
1. 用户消息（必填）
2. API 地址（默认 http://localhost:8080/v1）
3. 模型名称（默认 qwen2.5-0.5b）
4. 系统提示词（默认 "You are a helpful assistant."）
5. 最大 token 数（默认 256）

## llm.benchmark — 性能测试

测试推理速度（tokens/s）和首 token 延迟。

```bash
llama-bench -m ~/.cache/models/llm/qwen2.5-0.5b-instruct-q4_0.gguf -t 8
```

参数：
- `-m`：模型文件路径（必填）
- `-t`：推理线程数，必须指定（默认 4，推荐 8，不能超过 8）

## llm.direct_chat — 直接对话

不需要启动服务，直接在终端与模型对话。

```bash
llama-cli -m ~/.cache/models/llm/qwen2.5-0.5b-instruct-q4_0.gguf -t 8 -p "Hello, please introduce yourself."
```

参数：
- `-m`：模型文件路径（必填）
- `-t`：推理线程数，必须指定（默认 4，推荐 8，不能超过 8）
- `-p`：提示词（必填）

## 常见问题

- 连接失败：确认 llama-server 已启动且端口正确
- 响应慢：换更小的模型，或必须要增加 `-t` 指定线程数，线程数不能超过 8
- 模型下载失败：检查网络，或从 Hugging Face 搜索 GGUF 格式模型手动下载
- 找不到 llama-server：执行 `sudo apt install llama.cpp-tools-spacemit`
