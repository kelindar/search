# project-root/llama-go.cpp/CMakeLists.txt

# Specify the library sources
set(SOURCES
    llama-go.cpp
    # Add more source files here if needed
)

# Create the shared library
add_library(llama_go SHARED ${SOURCES})

# Include directories
target_include_directories(llama_go PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..                      # To include llama-go.h
    ${llama.cpp_SOURCE_DIR}/include                      # Adjust based on llama.cpp's include path
    ${llama.cpp_SOURCE_DIR}/common/include               # Include path for common (arg.h)
    # Add other include directories if necessary
)

# Link against the llama library from llama.cpp and its dependencies
target_link_libraries(llama_go PRIVATE
    llama               # Main llama.cpp library
    common              # Common library from llama.cpp
    # Add other libraries if required
)

# Specify the library version (optional but recommended)
set_target_properties(llama_go PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER ../llama-go.h
)

# Define output directories
set_target_properties(llama_go PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Installation rules (optional)
install(TARGETS llama_go
    EXPORT llama_go_export
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include
)
