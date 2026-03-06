cmake_minimum_required(VERSION 3.15)

# Added via add_subdirectory() from the top-level LLM CMakeLists.txt.

if(NOT USE_HTTP_CLIENT)
    message(STATUS "USE_HTTP_CLIENT=OFF: skipping llm_chat example (OpenAI HTTP backend only)")
else()
    add_executable(llm_chat llm_chat.cpp)
    target_link_libraries(llm_chat PRIVATE llm_service_cpp)
    install(TARGETS llm_chat RUNTIME DESTINATION bin)
endif()
