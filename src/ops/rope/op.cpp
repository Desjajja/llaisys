#include "op.hpp"
#include <cmath>
#include <complex>
#include <vector>

namespace llaisys::ops {
template <typename T>
void rope_impl(
    std::byte *out_base,
    const std::byte *in_base,
    size_t seqlen,
    size_t nhead,
    size_t d,
    size_t elem_size,
    const int64_t *pos_ids_ptr,
    const std::vector<double>& inv_freq
) {
    size_t half_d = d / 2;

    for (size_t s = 0; s < seqlen; ++s) {
        int64_t pos = pos_ids_ptr[s];
        for (size_t h = 0; h < nhead; ++h) {
            // Get the base pointer for the current vector [s, h, :]
            size_t vec_offset_bytes = elem_size * ((s * nhead * d) + (h * d));
            auto* current_out_vec = out_base + vec_offset_bytes;
            const auto* current_in_vec = in_base + vec_offset_bytes;

            // Loop through the first half of the dimensions
            for (size_t j = 0; j < half_d; ++j) {
                // Calculate the angle on the fly
                double angle = pos * inv_freq[j];

                // Create the rotation complex number
                auto rotation_complex = std::polar(1.0, angle);

                // Get the correct pair of values: x_j and x_{j + d/2}
                auto in_val_complex = std::complex<double>(
                    llaisys::utils::cast<double>(*reinterpret_cast<const T *>(current_in_vec + j * elem_size)),
                    llaisys::utils::cast<double>(*reinterpret_cast<const T *>(current_in_vec + (j + half_d) * elem_size))
                );

                // Apply the rotation
                auto mult_complex = in_val_complex * rotation_complex;

                // Write the results back to their correct positions
                *reinterpret_cast<T *>(current_out_vec + j * elem_size) = llaisys::utils::cast<T>(mult_complex.real());
                *reinterpret_cast<T *>(current_out_vec + (j + half_d) * elem_size) = llaisys::utils::cast<T>(mult_complex.imag());
            }
        }
    }
}


void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    auto *out_base = out->data();
    const auto *in_base = in->data();
    
    auto shape = in->shape();
    auto seqlen = shape[0];
    auto nhead = shape[1];
    auto d = shape[2];
    const auto elem_size = in->elementSize();
    
    // As confirmed before, pos_ids dtype must be handled correctly. Here we assume int64.
    const auto *pos_ids_ptr = reinterpret_cast<const int64_t *>(pos_ids->data());

    // FIX: Use the optimized inverse frequency calculation
    std::vector<double> inv_freq(d / 2);
    for (size_t i = 0; i < d / 2; ++i) {
        inv_freq[i] = 1.0 / std::pow(static_cast<double>(theta), static_cast<double>(2 * i) / d);
    }
    
    // Dispatch to the fixed implementation
    const auto dtype = in->dtype();
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rope_impl<float>(out_base, in_base, seqlen, nhead, d, elem_size, pos_ids_ptr, inv_freq);
    case LLAISYS_DTYPE_F16:
        return rope_impl<llaisys::fp16_t>(out_base, in_base, seqlen, nhead, d, elem_size, pos_ids_ptr, inv_freq);
    case LLAISYS_DTYPE_BF16:
        return rope_impl<llaisys::bf16_t>(out_base, in_base, seqlen, nhead, d, elem_size, pos_ids_ptr, inv_freq);
    case LLAISYS_DTYPE_I8:
        return rope_impl<int8_t>(out_base, in_base, seqlen, nhead, d, elem_size, pos_ids_ptr, inv_freq);
    case LLAISYS_DTYPE_I16:
        return rope_impl<int16_t>(out_base, in_base, seqlen, nhead, d, elem_size, pos_ids_ptr, inv_freq);
    case LLAISYS_DTYPE_I32:
        return rope_impl<int32_t>(out_base, in_base, seqlen, nhead, d, elem_size, pos_ids_ptr, inv_freq);
    case LLAISYS_DTYPE_I64:
        return rope_impl<int64_t>(out_base, in_base, seqlen, nhead, d, elem_size, pos_ids_ptr, inv_freq);
    case LLAISYS_DTYPE_U8:
        return rope_impl<uint8_t>(out_base, in_base, seqlen, nhead, d, elem_size, pos_ids_ptr, inv_freq);
    case LLAISYS_DTYPE_U16:
        return rope_impl<uint16_t>(out_base, in_base, seqlen, nhead, d, elem_size, pos_ids_ptr, inv_freq);
    case LLAISYS_DTYPE_U32:
        return rope_impl<uint32_t>(out_base, in_base, seqlen, nhead, d, elem_size, pos_ids_ptr, inv_freq);
    case LLAISYS_DTYPE_U64:
        return rope_impl<uint64_t>(out_base, in_base, seqlen, nhead, d, elem_size, pos_ids_ptr, inv_freq);
    default:
        throw std::runtime_error("rope: unsupported or non-numeric dtype");
    }
}
} // namespace llaisys::ops
