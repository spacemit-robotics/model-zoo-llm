cmake_minimum_required(VERSION 3.15)
project(llm_service_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(USE_HTTP_CLIENT "Enable HTTP client (OpenAI API mode)" ON)
option(BUILD_SHARED_LIB "Build shared library" ON)
option(BUILD_LLM_EXAMPLES "Build example programs under example/" ON)

set(SOURCES
    src/llm_service.cpp
    src/llm_backend_factory.cpp
    src/backends/openai_backend.cpp
)

# Build library
if(BUILD_SHARED_LIB)
    add_library(llm_service_cpp SHARED ${SOURCES})
    set_target_properties(llm_service_cpp PROPERTIES
        VERSION 1.0.0
        SOVERSION 1
    )
else()
    add_library(llm_service_cpp STATIC ${SOURCES})
endif()

# Alias so consumers can use target name "llm"
add_library(llm ALIAS llm_service_cpp)

# Public include directories (exported to consumers)
target_include_directories(llm_service_cpp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Private include for internal src headers
target_include_directories(llm_service_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# HTTP client (libcurl + nlohmann_json)
if(USE_HTTP_CLIENT)
    find_package(CURL)
    # Check if nlohmann_json target already exists (e.g., from MCP's FetchContent)
    if(TARGET nlohmann_json::nlohmann_json)
        set(nlohmann_json_available TRUE)
    else()
        find_package(nlohmann_json QUIET)
        set(nlohmann_json_available ${nlohmann_json_FOUND})
    endif()
    if(CURL_FOUND AND nlohmann_json_available)
        target_compile_definitions(llm_service_cpp PRIVATE USE_HTTP_CLIENT)
        target_link_libraries(llm_service_cpp PRIVATE CURL::libcurl)
        target_link_libraries(llm_service_cpp PRIVATE nlohmann_json::nlohmann_json)
    else()
        message(WARNING "USE_HTTP_CLIENT=ON but CURL/nlohmann_json not found. Building without HTTP client support.")
    endif()
endif()

# Compiler-specific options
if(MSVC)
    target_compile_options(llm_service_cpp PRIVATE /W4)
else()
    target_compile_options(llm_service_cpp PRIVATE -Wall -Wextra -O3)
endif()

# Install
install(TARGETS llm_service_cpp
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES
    include/llm_service.h
    DESTINATION include
)

#
# Examples (built as part of the main project)
#
if(BUILD_LLM_EXAMPLES)
    add_subdirectory(example/cpp)
endif()
