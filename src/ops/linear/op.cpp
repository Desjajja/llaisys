#include "op.hpp"
#include "../../utils/types.hpp"

namespace llaisys::ops {

// Compute Y = X * W^T + b
// Shapes:
//   X: [B, In]
//   W: [Out, In]
//   b (optional): [Out]
//   Y: [B, Out]
// Assumes no broadcasting beyond optional bias add.
template <typename T>
void linear_impl(std::byte *out_base,
                 const std::byte *in_base,
                 const std::byte *w_base,
                 const std::byte *bias_base,
                 size_t batch_size,
                 size_t out_features,
                 size_t in_features,
                 size_t elem_size,
                 ptrdiff_t in_col_stride_bytes,
                 ptrdiff_t in_batch_stride_bytes,
                 ptrdiff_t w_row_stride_bytes,      // stride between rows (output neurons) in W
                 ptrdiff_t out_col_stride_bytes,    // stride between output features
                 ptrdiff_t out_batch_stride_bytes)  // stride between output batches
{
    using llaisys::utils::cast;

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t o = 0; o < out_features; ++o) {
            double acc = 0.0;
            // Dot product of input row b with weight row o
            for (size_t i = 0; i < in_features; ++i) {
                const auto in_offset = static_cast<ptrdiff_t>(b * in_batch_stride_bytes + i * in_col_stride_bytes);
                const auto w_offset  = static_cast<ptrdiff_t>(o * w_row_stride_bytes + i * elem_size);
                const T in_val = *reinterpret_cast<const T *>(in_base + in_offset);
                const T w_val  = *reinterpret_cast<const T *>(w_base + w_offset);
                acc += static_cast<double>(cast<float>(in_val)) * static_cast<double>(cast<float>(w_val));
            }

            float result = static_cast<float>(acc);
            if (bias_base) {
                const auto bias_offset = static_cast<ptrdiff_t>(o * elem_size);
                const T b_val = *reinterpret_cast<const T *>(bias_base + bias_offset);
                result += cast<float>(b_val);
            }

            const auto out_offset = static_cast<ptrdiff_t>(b * out_batch_stride_bytes + o * out_col_stride_bytes);
            // *reinterpret_cast<float *>(out_base + out_offset) = result; // Output stored as F32
            auto *dst = reinterpret_cast<T *>(out_base + out_offset);
            *dst = llaisys::utils::cast<T>(result);
        }
    }
}

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    const auto dtype = weight->dtype();
    const auto elem_size = static_cast<size_t>(weight->elementSize());

    // Weight shape: [Out, In]
    const auto &w_shape = weight->shape();
    if (w_shape.size() != 2)
        throw std::runtime_error("weight must be 2D");
    const size_t out_features = w_shape[0];
    const size_t in_features  = w_shape[1];

    // Input shape: [B, In]
    const auto &in_shape = in->shape();
    if (in_shape.size() != 2)
        throw std::runtime_error("input must be 2D");
    const size_t batch_size = in_shape[0];
    if (in_shape[1] != in_features)
        throw std::runtime_error("input feature dim mismatch with weight");

    // Output shape (expected): [B, Out]
    const auto &out_shape = out->shape();
    if (out_shape.size() != 2 || out_shape[0] != batch_size || out_shape[1] != out_features)
        throw std::runtime_error("output shape mismatch");

    if (bias) {
        const auto &b_shape = bias->shape();
        if (!(b_shape.size() == 1 && b_shape[0] == out_features))
            throw std::runtime_error("bias shape must be [Out]");
        if (bias->elementSize() != elem_size)
            throw std::runtime_error("bias dtype mismatch");
    }

    // Dtype size check
    if (elem_size != static_cast<size_t>(llaisys::utils::dsize(dtype)))
        throw std::runtime_error("element size does not match dtype size");

    const std::byte *w_base    = weight->data();
    const std::byte *in_base   = in->data();
    const std::byte *bias_base = bias ? bias->data() : nullptr;
    std::byte *out_base        = out->data();

    const auto w_strides   = weight->strides(); // [Out, In]
    const auto in_strides  = in->strides();     // [B, In]
    const auto out_strides = out->strides();    // [B, Out]

    // Strides (in elements) -> bytes
    const ptrdiff_t w_row_stride_bytes     = static_cast<ptrdiff_t>(w_strides[0])   * static_cast<ptrdiff_t>(elem_size);
    const ptrdiff_t in_col_stride_bytes    = static_cast<ptrdiff_t>(in_strides[1])  * static_cast<ptrdiff_t>(elem_size);
    const ptrdiff_t in_batch_stride_bytes  = static_cast<ptrdiff_t>(in_strides[0])  * static_cast<ptrdiff_t>(elem_size);
    const ptrdiff_t out_col_stride_bytes   = static_cast<ptrdiff_t>(out_strides[1]) * static_cast<ptrdiff_t>(elem_size);
    const ptrdiff_t out_batch_stride_bytes = static_cast<ptrdiff_t>(out_strides[0]) * static_cast<ptrdiff_t>(elem_size);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return linear_impl<float>(out_base, in_base, w_base, bias_base,
                                  batch_size, out_features, in_features, elem_size,
                                  in_col_stride_bytes, in_batch_stride_bytes,
                                  w_row_stride_bytes, out_col_stride_bytes, out_batch_stride_bytes);
    case LLAISYS_DTYPE_F16:
        return linear_impl<llaisys::fp16_t>(out_base, in_base, w_base, bias_base,
                                            batch_size, out_features, in_features, elem_size,
                                            in_col_stride_bytes, in_batch_stride_bytes,
                                            w_row_stride_bytes, out_col_stride_bytes, out_batch_stride_bytes);
    case LLAISYS_DTYPE_BF16:
        return linear_impl<llaisys::bf16_t>(out_base, in_base, w_base, bias_base,
                                            batch_size, out_features, in_features, elem_size,
                                            in_col_stride_bytes, in_batch_stride_bytes,
                                            w_row_stride_bytes, out_col_stride_bytes, out_batch_stride_bytes);
    case LLAISYS_DTYPE_I8:
        return linear_impl<int8_t>(out_base, in_base, w_base, bias_base,
                                   batch_size, out_features, in_features, elem_size,
                                   in_col_stride_bytes, in_batch_stride_bytes,
                                   w_row_stride_bytes, out_col_stride_bytes, out_batch_stride_bytes);
    case LLAISYS_DTYPE_I16:
        return linear_impl<int16_t>(out_base, in_base, w_base, bias_base,
                                    batch_size, out_features, in_features, elem_size,
                                    in_col_stride_bytes, in_batch_stride_bytes,
                                    w_row_stride_bytes, out_col_stride_bytes, out_batch_stride_bytes);
    case LLAISYS_DTYPE_I32:
        return linear_impl<int32_t>(out_base, in_base, w_base, bias_base,
                                    batch_size, out_features, in_features, elem_size,
                                    in_col_stride_bytes, in_batch_stride_bytes,
                                    w_row_stride_bytes, out_col_stride_bytes, out_batch_stride_bytes);
    case LLAISYS_DTYPE_I64:
        return linear_impl<int64_t>(out_base, in_base, w_base, bias_base,
                                    batch_size, out_features, in_features, elem_size,
                                    in_col_stride_bytes, in_batch_stride_bytes,
                                    w_row_stride_bytes, out_col_stride_bytes, out_batch_stride_bytes);
    case LLAISYS_DTYPE_U8:
        return linear_impl<uint8_t>(out_base, in_base, w_base, bias_base,
                                    batch_size, out_features, in_features, elem_size,
                                    in_col_stride_bytes, in_batch_stride_bytes,
                                    w_row_stride_bytes, out_col_stride_bytes, out_batch_stride_bytes);
    case LLAISYS_DTYPE_U16:
        return linear_impl<uint16_t>(out_base, in_base, w_base, bias_base,
                                     batch_size, out_features, in_features, elem_size,
                                     in_col_stride_bytes, in_batch_stride_bytes,
                                     w_row_stride_bytes, out_col_stride_bytes, out_batch_stride_bytes);
    case LLAISYS_DTYPE_U32:
        return linear_impl<uint32_t>(out_base, in_base, w_base, bias_base,
                                     batch_size, out_features, in_features, elem_size,
                                     in_col_stride_bytes, in_batch_stride_bytes,
                                     w_row_stride_bytes, out_col_stride_bytes, out_batch_stride_bytes);
    case LLAISYS_DTYPE_U64:
        return linear_impl<uint64_t>(out_base, in_base, w_base, bias_base,
                                     batch_size, out_features, in_features, elem_size,
                                     in_col_stride_bytes, in_batch_stride_bytes,
                                     w_row_stride_bytes, out_col_stride_bytes, out_batch_stride_bytes);
    default:
        throw std::runtime_error("linear: unsupported or non-numeric dtype");
    }
}

}