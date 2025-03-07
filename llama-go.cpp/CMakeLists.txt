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
    ${CMAKE_CURRENT_SOURCE_DIR}/..                            # To include llama-go.h
    ${llama.cpp_SOURCE_DIR}/include                            # Adjust based on llama.cpp's include path
    ${llama.cpp_SOURCE_DIR}/common                     # Include path for common (arg.h)
    # Add other include directories if necessary
)

# Define preprocessor macros for exporting symbols
target_compile_definitions(llama_go PRIVATE
    LLAMA_SHARED   # Indicates that we are building a shared library
    LLAMA_BUILD    # Indicates that we are building the llama library (exporting symbols)
)

# Link against the static llama and common libraries from llama.cpp
target_link_libraries(llama_go PRIVATE
    llama       # Static llama library
    common      # Static common library
    # Add other libraries if required
)

# Specify the library version
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

# Installation rules 
install(TARGETS llama_go
    EXPORT llama_go_export
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include
)
