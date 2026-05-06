#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;

// comma separated list of all types
std::string common_speculative_type_name_str();

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// check if the llama_context is compatible for speculative decoding
// note: clears the memory of the context
bool common_speculative_is_compat(llama_context * ctx_tgt);

common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt);

void common_speculative_free(common_speculative * spec);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt);

// set target-side sequence id used by implementations that read from the target's KV memory
// (currently only used by the MTP implementation; safe no-op for others)
void common_speculative_set_seq_id(common_speculative * spec, llama_seq_id seq_id);

// Set the output index in the target's most recent decode whose embeddings should be read
// as h_prev for the next MTP draft. -1 means "last output" (default).
// In speculative verification, after partial draft acceptance the last batch output corresponds
// to a rejected draft; the correct h_prev is at the last *accepted* batch index.
// Safe no-op for non-MTP implementations.
void common_speculative_set_h_idx(common_speculative * spec, int batch_idx);

// sample up to n_draft tokens and add them to the batch using the draft model
llama_tokens common_speculative_draft(
                     common_speculative * spec,
        const common_params_speculative & params,
                     const llama_tokens & prompt,
                            llama_token   id_last);

// informs the speculative decoder that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, uint16_t n_accepted);

// After target sample/accept, submit MTP work for the next iteration so it can overlap
// server bookkeeping until the next common_speculative_draft() (pipeline depth-2, no optimistic token).
// Safe no-op for non-MTP implementations.
void common_speculative_prepare_next(common_speculative * spec, llama_token id_last);

// Drain any pending async draft from a previous prepare_next() and discard the result.
// MUST be called before the host mutates target KV in a way that would invalidate the
// snapshot (e.g. slot stop / release / new request seq_rm). Safe no-op when nothing is pending.
void common_speculative_cancel(common_speculative * spec);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);
