/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "llm_backend_factory.h"

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "backends/openai_backend.h"

namespace spacemit_llm {

static std::map<std::string, LLMBackendFactory::BackendCreator>& get_custom_backends() {
    static std::map<std::string, LLMBackendFactory::BackendCreator> custom_backends;
    return custom_backends;
}

static std::mutex& get_registry_mutex() {
    static std::mutex registry_mutex;
    return registry_mutex;
}

std::unique_ptr<LLMBackend> LLMBackendFactory::create(const BackendConfig& config) {
    switch (config.type) {
        case BackendConfig::Type::OPENAI_API: {
            auto backend = std::make_unique<OpenAIBackend>(config);
            if (backend->initialize()) {
                return backend;
            }
            return nullptr;
        }
        case BackendConfig::Type::CUSTOM: {
            std::lock_guard<std::mutex> lock(get_registry_mutex());
            auto& backends = get_custom_backends();
            auto it = backends.find(config.custom.backend_name);
            if (it != backends.end()) {
                return it->second(config);
            }
            return nullptr;
        }
        default:
            return nullptr;
    }
}

std::unique_ptr<LLMBackend> LLMBackendFactory::create_from_string(
    const std::string& backend_type,
    const std::string& config_str
) {
    (void)config_str;  // 预留：后续支持从字符串(JSON/key=value)解析配置

    BackendConfig config;
    if (backend_type == "openai" || backend_type == "openai_api") {
        config.type = BackendConfig::Type::OPENAI_API;
        config.openai.api_base = "http://localhost:8000/v1";
        config.openai.api_key = "EMPTY";
        config.openai.model = "gpt-4o-mini";
    } else {
        config.type = BackendConfig::Type::CUSTOM;
        config.custom.backend_name = backend_type;
    }

    return create(config);
}

void LLMBackendFactory::register_backend(const std::string& name, BackendCreator creator) {
    std::lock_guard<std::mutex> lock(get_registry_mutex());
    get_custom_backends()[name] = creator;
}

std::vector<std::string> LLMBackendFactory::list_available_backends() {
    std::vector<std::string> backends;
    backends.push_back("openai_api");

    std::lock_guard<std::mutex> lock(get_registry_mutex());
    for (const auto& pair : get_custom_backends()) {
        backends.push_back(pair.first);
    }

    return backends;
}

}  // namespace spacemit_llm
