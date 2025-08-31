#include "op.hpp"
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

namespace llaisys::ops {

// Helper function for numerically stable softmax
void softmax(const std::vector<float> &v, std::vector<float> &v_exp) {
    if (v.empty()) {
        return;
    }
    float max_val = *std::max_element(v.begin(), v.end());
    float sum = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        v_exp[i] = std::exp(v[i] - max_val);
        sum += v_exp[i];
    }
    for (size_t i = 0; i < v_exp.size(); ++i) {
        v_exp[i] /= sum;
    }
}

template <typename T>
void self_attn_impl(
    size_t qlen,
    size_t kvlen, // total_len from the K/V cache
    size_t nhead,
    size_t nkvhead,
    size_t d,
    size_t dv,
    size_t elem_size,
    const std::byte *q_base,
    const std::byte *k_base,
    const std::byte *v_base,
    std::byte *attn_base,
    float scale) {

    const size_t heads_per_kv = nhead / nkvhead;
    const size_t kv_cache_len = kvlen - qlen;

    // Loop over each query token in the current batch
    for (size_t s = 0; s < qlen; ++s) {
        // The absolute position of the current query in the full sequence
        const size_t absolute_pos = kv_cache_len + s;

        // Loop over each query head
        for (size_t h = 0; h < nhead; ++h) {
            // Find the corresponding key/value head for the current query head (for GQA)
            const size_t hk = h / heads_per_kv;

            // --- 1. Calculate Attention Scores (Q * K^T * scale) ---
            // For causal attention, we only attend to keys up to the current absolute position.
            const size_t attention_span = absolute_pos + 1;
            std::vector<float> qk_prod(attention_span);

            auto q_offset_byte = static_cast<ptrdiff_t>(((s * nhead * d) + (h * d)) * elem_size);
            
            for (size_t s_k = 0; s_k < attention_span; ++s_k) {
                auto k_offset_byte = static_cast<ptrdiff_t>(((s_k * nkvhead * d) + (hk * d)) * elem_size);
                float current_qk_prod = 0.0f;
                for (size_t j = 0; j < d; ++j) {
                    auto q_val = llaisys::utils::cast<float>(*reinterpret_cast<const T *>(q_base + q_offset_byte + j * elem_size));
                    auto k_val = llaisys::utils::cast<float>(*reinterpret_cast<const T *>(k_base + k_offset_byte + j * elem_size));
                    current_qk_prod += q_val * k_val;
                }
                qk_prod[s_k] = current_qk_prod * scale;
            }

            // --- 2. Apply Causal Softmax ---
            std::vector<float> qk_logits(attention_span);
            softmax(qk_prod, qk_logits);

            // --- 3. Calculate Final Output (Softmax_Scores * V) ---
            auto attn_offset_byte = static_cast<ptrdiff_t>(((s * nhead * dv) + (h * dv)) * elem_size);
            
            for (size_t j = 0; j < dv; ++j) {
                float acc_val = 0.0;
                for (size_t s_v = 0; s_v < attention_span; ++s_v) {
                    auto v_offset_byte = static_cast<ptrdiff_t>(((s_v * nkvhead * dv) + (hk * dv)) * elem_size);
                    auto v_val = llaisys::utils::cast<float>(*reinterpret_cast<const T *>(v_base + v_offset_byte + j * elem_size));
                    acc_val += qk_logits[s_v] * v_val;
                }
                *reinterpret_cast<T *>(attn_base + attn_offset_byte + j * elem_size) = llaisys::utils::cast<T>(acc_val);
            }
        }
    }
}

// Public-facing wrapper function
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    size_t qlen = q->shape()[0];
    size_t kvlen = k->shape()[0]; // This is `total_len`
    size_t nhead = q->shape()[1];
    size_t nkvhead = k->shape()[1];
    size_t d = q->shape()[2];
    size_t dv = v->shape()[2];
    size_t elem_size = q->elementSize();
    const auto *q_base = q->data();
    const auto *k_base = k->data();
    const auto *v_base = v->data();
    auto *attn_base = attn_val->data();
    const auto dtype = attn_val->dtype();

    // Dispatch to the correct templated implementation based on data type
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        self_attn_impl<float>(qlen, kvlen, nhead, nkvhead, d, dv, elem_size, q_base, k_base, v_base, attn_base, scale);
        break;
    case LLAISYS_DTYPE_F16:
        self_attn_impl<llaisys::fp16_t>(qlen, kvlen, nhead, nkvhead, d, dv, elem_size, q_base, k_base, v_base, attn_base, scale);
        break;
    case LLAISYS_DTYPE_BF16:
        self_attn_impl<llaisys::bf16_t>(qlen, kvlen, nhead, nkvhead, d, dv, elem_size, q_base, k_base, v_base, attn_base, scale);
        break;

    default:
        throw std::runtime_error("self_attention: unsupported or non-numeric dtype");
    }
}

} // namespace llaisys::ops