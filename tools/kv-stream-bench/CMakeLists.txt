set(TARGET llama-kv-stream-bench)

find_package(CUDAToolkit REQUIRED)

add_executable(${TARGET}
    main.cpp
    kv-stream-bench-config.cpp
    kv-stream-bench-config.h)

target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_link_libraries(${TARGET} PRIVATE CUDA::cudart)

if (LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
