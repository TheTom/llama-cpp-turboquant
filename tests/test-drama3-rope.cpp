#include "../src/llama-model.h"

#include <cstdlib>
#include <memory>

static void check(llm_arch arch, const std::array<int32_t, 4> & sections, llama_rope_type expected) {
    std::unique_ptr<llama_model> model(llama_model_create(arch, llama_model_default_params()));
    model->hparams.rope_sections = sections;
    if (llama_model_rope_type(model.get()) != expected) {
        std::abort();
    }
}

int main() {
    check(LLM_ARCH_QWEN35MOE, { 0, 0, 0, 0 }, LLAMA_ROPE_TYPE_NEOX);
    check(LLM_ARCH_QWEN35MOE, { 11, 11, 10, 0 }, LLAMA_ROPE_TYPE_IMROPE);
    check(LLM_ARCH_QWEN35MOE, { 32, 0, 0, 0 }, LLAMA_ROPE_TYPE_IMROPE);
    check(LLM_ARCH_QWEN35, { 0, 0, 0, 0 }, LLAMA_ROPE_TYPE_IMROPE);
    check(LLM_ARCH_QWEN3NEXT, { 0, 0, 0, 0 }, LLAMA_ROPE_TYPE_NEOX);
    return 0;
}
