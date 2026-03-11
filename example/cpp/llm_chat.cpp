/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Unified example for OpenAI-compatible LLM services (e.g. llama.cpp llama-server).
 * Use with any model: Qwen, DeepSeek, etc. by passing model name as argument.
 */

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <cstdlib>

#include "llm_service.h"

static void print_usage(const char* prog) {
    std::cerr << "Usage:\n";
    std::cerr << "  " << prog
            << " \"<user_text>\" [api_base] [model] [prompt] [max_tokens]\n\n";
    std::cerr << "Examples (OpenAI-compatible server, e.g. llama.cpp llama-server):\n";
    std::cerr << "  " << prog << " \"Hello\" http://localhost:8080/v1 qwen2.5-0.5b"
            << " \"You are a helpful assistant.\" 256\n";
    std::cerr << "  " << prog << " \"Write a haiku\" http://localhost:8080/v1 deepseek-r1-1.5b"
            << " \"You are a helpful assistant.\" 256\n";
}

static std::optional<int> try_parse_int(const std::string& s) {
    try {
        size_t idx = 0;
        int v = std::stoi(s, &idx);
        if (idx != s.size()) {
            return std::nullopt;
        }
        return v;
    } catch (...) {
        return std::nullopt;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string user_text = argv[1];
    const std::string api_base = (argc > 2) ? argv[2] : "http://localhost:8080/v1";
    const std::string model = (argc > 3) ? argv[3] : "qwen2.5-0.5b";
    std::string prompt = "You are a helpful assistant.";
    int max_tokens = 256;

    // Optional args: [prompt] [max_tokens]; if last arg is int, treat as max_tokens
    if (argc > 4) {
        const std::string last = argv[argc - 1];
        if (auto v = try_parse_int(last)) {
            max_tokens = *v;
            if (argc > 5) {
                prompt = argv[4];
            }
        } else {
            prompt = argv[4];
        }
    }

    try {
        const char* key_env = std::getenv("OPENAI_API_KEY");
        std::string api_key = key_env ? key_env : "";
        spacemit_llm::LLMService llm(model, api_base, api_key, prompt, max_tokens);

        std::cout << "=== LLM (OpenAI-compatible) ===\n";
        std::cout << "API base: " << api_base << "\n";
        std::cout << "Model:    " << model << "\n";
        std::cout << "Prompt:   " << prompt << "\n";
        std::cout << "Input:    " << user_text << "\n\n";

        std::cout << "=== Response (stream) ===\n";

        std::mutex mtx;
        std::condition_variable cv;
        bool finished = false;
        std::string error;

        const auto start = std::chrono::high_resolution_clock::now();
        llm.complete_stream(
            user_text,
            [&](const std::string& chunk, bool is_finished,
                const std::string& err) -> bool {
                std::lock_guard<std::mutex> lock(mtx);
                if (!err.empty()) {
                    error = err;
                    finished = true;
                    cv.notify_one();
                    return false;
                }
                if (!chunk.empty()) {
                    std::cout << chunk << std::flush;
                }
                if (is_finished) {
                    finished = true;
                    cv.notify_one();
                }
                return true;
            });

        constexpr int timeout_seconds = 180;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait_for(lock, std::chrono::seconds(timeout_seconds),
                        [&] { return finished; });
        }

        if (!error.empty()) {
            std::cerr << "\n\nError: " << error << "\n";
            return 2;
        }

        std::cout << "\n\n=== Metrics ===\n";
        auto metrics = llm.get_metrics();
        std::cout << "Requests:       " << metrics.total_requests << "\n";
        if (metrics.last_ttft_ms > 0) {
            std::cout << "TTFT:           " << metrics.last_ttft_ms << " ms\n";
        }
        if (metrics.last_output_tokens >= 0) {
            std::cout << "Tokens:         out=" << metrics.last_output_tokens << "\n";
            if (metrics.last_tokens_per_second > 0) {
                std::cout << "Throughput:     " << metrics.last_tokens_per_second
                        << " tokens/s\n";
            }
        }
        std::cout << "E2E latency:    " << metrics.last_latency_ms << " ms (last)  "
                << metrics.avg_latency_ms << " ms (avg)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
