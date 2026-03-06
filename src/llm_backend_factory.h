/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef LLM_BACKEND_FACTORY_H
#define LLM_BACKEND_FACTORY_H

#include <memory>
#include <string>
#include <vector>

#include "llm_backend.h"

namespace spacemit_llm {

/**
 * LLM Backend Factory
 * 支持注册自定义后端
 */
class LLMBackendFactory {
public:
    static std::unique_ptr<LLMBackend> create(const BackendConfig& config);

    static std::unique_ptr<LLMBackend> create_from_string(
        const std::string& backend_type,
        const std::string& config_str);

    using BackendCreator = std::function<std::unique_ptr<LLMBackend>(const BackendConfig&)>;
    static void register_backend(const std::string& name, BackendCreator creator);

    static std::vector<std::string> list_available_backends();
};

}  // namespace spacemit_llm

#endif  // LLM_BACKEND_FACTORY_H
