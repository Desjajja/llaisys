#include "op.hpp"
#include "../../utils/types.hpp"
#include <cmath>
#include <vector>

namespace llaisys::ops {

template<typename T>
void rms_norm_impl(
    std::byte *out_base,
    const std::byte *in_base,
    const std::byte *w_base,
    const size_t elem_size,
    size_t in_batch_stride,
    size_t in_row_num,
    size_t in_col_num,
    size_t d,
    float eps
) {
    auto in_batch_stride_bytes = in_batch_stride * elem_size;
    std::vector<float> in_row_vals(in_col_num);

    for (size_t row = 0; row < in_row_num; ++row) {

        float acc_square = 0.0f;
        for (size_t col = 0; col < in_col_num; ++col) {
            const auto in_offset = static_cast<ptrdiff_t>(row * in_batch_stride_bytes + col * elem_size);
            const auto in_val = llaisys::utils::cast<float>(*reinterpret_cast<const T *>(in_base + in_offset));
            in_row_vals[col] = in_val;
            acc_square += in_val * in_val;
        }
        const float rsqrt_denominator = 1.0f / sqrt(acc_square / d + eps);

        for (size_t col = 0; col < in_col_num; ++col) {
            const auto w_offset     = static_cast<ptrdiff_t>(col * elem_size);
            const auto o_offset     = static_cast<ptrdiff_t>(row * in_batch_stride_bytes + col * elem_size);
            const auto w_val = llaisys::utils::cast<float>(*reinterpret_cast<const T *>(w_base + w_offset));
            const float normalized_val = (in_row_vals[col] * rsqrt_denominator) * w_val;
            auto *dst = reinterpret_cast<T *>(out_base + o_offset);
            *dst = llaisys::utils::cast<T>(normalized_val);
        }
    }
}

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    auto *out_base      = out->data();
    const auto in_base  = in->data();
    const auto w_base   = weight->data();
    const auto elem_size = in->elementSize();

    size_t in_batch_stride = in->strides()[0];
    size_t in_row_num = in->shape()[0];
    size_t in_col_num = in->shape()[1];
    size_t d = weight->shape()[0];
    if (d != in_col_num) {
        throw std::runtime_error("weight's shape mismatch with input!");
    }

    const auto dtype = in->dtype();
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_impl<float>(out_base, in_base, w_base, elem_size,
                                  in_batch_stride, in_row_num, in_col_num, d, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_impl<llaisys::fp16_t>(out_base, in_base, w_base, elem_size,
                                  in_batch_stride, in_row_num, in_col_num, d, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_impl<llaisys::bf16_t>(out_base, in_base, w_base, elem_size,
                                  in_batch_stride, in_row_num, in_col_num, d, eps);
    case LLAISYS_DTYPE_I8:
        return rms_norm_impl<int8_t>(out_base, in_base, w_base, elem_size,
                                  in_batch_stride, in_row_num, in_col_num, d, eps);
    case LLAISYS_DTYPE_I16:
        return rms_norm_impl<int16_t>(out_base, in_base, w_base, elem_size,
                                  in_batch_stride, in_row_num, in_col_num, d, eps);
    case LLAISYS_DTYPE_I32:
        return rms_norm_impl<int32_t>(out_base, in_base, w_base, elem_size,
                                  in_batch_stride, in_row_num, in_col_num, d, eps);
    case LLAISYS_DTYPE_I64:
        return rms_norm_impl<int64_t>(out_base, in_base, w_base, elem_size,
                                  in_batch_stride, in_row_num, in_col_num, d, eps);
    case LLAISYS_DTYPE_U8:
        return rms_norm_impl<uint8_t>(out_base, in_base, w_base, elem_size,
                                  in_batch_stride, in_row_num, in_col_num, d, eps);
    case LLAISYS_DTYPE_U16:
        return rms_norm_impl<uint16_t>(out_base, in_base, w_base, elem_size,
                                  in_batch_stride, in_row_num, in_col_num, d, eps);
    case LLAISYS_DTYPE_U32:
        return rms_norm_impl<uint32_t>(out_base, in_base, w_base, elem_size,
                                  in_batch_stride, in_row_num, in_col_num, d, eps);
    case LLAISYS_DTYPE_U64:
        return rms_norm_impl<uint64_t>(out_base, in_base, w_base, elem_size,
                                  in_batch_stride, in_row_num, in_col_num, d, eps);
    default:
        throw std::runtime_error("linear: unsupported or non-numeric dtype");
    }

}
} // namespace llaisys::ops
