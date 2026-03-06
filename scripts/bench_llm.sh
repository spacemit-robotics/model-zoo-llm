#!/usr/bin/env bash

MODEL_PATH="$1"
PORT="${PORT:-8080}"
THREADS="${LLM_THREADS:-8}"

if [ -z "$MODEL_PATH" ]; then
  echo "Usage: $0 /path/to/model.gguf"
  exit 1
fi

# 启动 llama-server（后台运行）
llama-server -m "$MODEL_PATH" -t "$THREADS" --port "$PORT" >/tmp/llama_server.log 2>&1 &
SERVER_PID=$!

# 等待服务就绪（最多 300 秒，模型越大越慢）
echo "waiting for llama-server to be ready ..."

sleep 3

READY=0
for i in $(seq 1 300); do
  # curl 默认对 4xx/5xx 也返回 0，需要用 -f 让 HTTP 错误变为非 0
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 2
done

if [ "$READY" -ne 1 ]; then
  echo "llama-server not ready after 300s, aborting."
  kill "$SERVER_PID" >/dev/null 2>&1 || true
  exit 1
fi

sleep 1

PROMPT="You are a helpful assistant. Please answer the following question in detail. \
Explain the basic concepts of large language models, including tokens, context \
length, and how they generate text. Make the explanation easy to understand."

# 优先用当前仓库编译的 llm_chat，否则用 PATH 中的 llm_chat
if [ -x "./build/example/cpp/llm_chat" ]; then
  LLM_CHAT="./build/example/cpp/llm_chat"
else
  LLM_CHAT="llm_chat"
fi

echo "running llm_chat benchmark ..."
# 预热一次，丢弃这次结果
"$LLM_CHAT" "warmup" \
  "http://127.0.0.1:${PORT}/v1" \
  "bench-model" \
  "You are a helpful assistant." \
  16 >/dev/null 2>&1 || true

# 真正测一次，记录 Metrics
"$LLM_CHAT" "$PROMPT" \
  "http://127.0.0.1:${PORT}/v1" \
  "bench-model" \
  "You are a helpful assistant." \
  128

kill "$SERVER_PID" >/dev/null 2>&1 || true
