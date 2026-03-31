#pragma once
// TurboQuant InnerQ per-channel equalization - cross-TU shared state

#define INNERQ_MAX_CHANNELS 128

// DLL export/import macro
#if defined(_WIN32)
#  if defined(GGML_BACKEND_BUILD)
#    define TURBO_INNERQ_API __declspec(dllexport)
#  else
#    define TURBO_INNERQ_API __declspec(dllimport)
#  endif
#else
#  define TURBO_INNERQ_API
#endif

// Host-side shared state (defined in turbo-innerq.cu)
extern TURBO_INNERQ_API bool  g_innerq_finalized;
extern TURBO_INNERQ_API float g_innerq_scale_inv_host[INNERQ_MAX_CHANNELS];

TURBO_INNERQ_API void turbo_innerq_publish(const float * scale_inv, int group_size);
TURBO_INNERQ_API bool turbo_innerq_needs_tensor_update(void);
TURBO_INNERQ_API void turbo_innerq_mark_tensor_updated(void);
