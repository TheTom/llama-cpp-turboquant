set(TARGET llama-triattention-calibrate)
add_executable(${TARGET} triattention-calibrate.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
# Reuses the .triattention binary format constants/structs from the internal
# TriAttention runtime header rather than redefining them, to keep the writer
# here and the loader in src/llama-triattention.cpp from drifting apart.
target_include_directories(${TARGET} PRIVATE ../../src)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
