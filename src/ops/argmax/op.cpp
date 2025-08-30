#include "op.hpp"
#include "../../utils/types.hpp"

namespace llaisys::ops {

using namespace llaisys::utils;

template <typename T>
struct ArgmaxAdapter {
    static inline float to_float(T v) { return static_cast<float>(v); }
    static inline T from_float(float v) { return static_cast<T>(v); }
};

template <>
struct ArgmaxAdapter<fp16_t> {
    static inline float to_float(fp16_t v) { return _f16_to_f32(v); }
    static inline fp16_t from_float(float v) { return _f32_to_f16(v); }
};

template <>
struct ArgmaxAdapter<bf16_t> {
    static inline float to_float(bf16_t v) { return _bf16_to_f32(v); }
    static inline bf16_t from_float(float v) { return _f32_to_bf16(v); }
};

template <typename T>
void argmax_impl(tensor_t max_idx, tensor_t max_val, const T* data, size_t n) {
    float max_f = ArgmaxAdapter<T>::to_float(data[0]);
    size_t idx = 0;
    for (size_t i = 1; i < n; ++i) {
        float v = ArgmaxAdapter<T>::to_float(data[i]);
        if (v > max_f) {
            max_f = v;
            idx = i;
        }
    }
    *reinterpret_cast<int64_t*>(max_idx->data()) = static_cast<int64_t>(idx);
    *reinterpret_cast<T*>(max_val->data()) = ArgmaxAdapter<T>::from_float(max_f);
}

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    size_t n = vals->numel();
    if (n == 0) throw std::runtime_error("argmax: empty tensor");

    switch (vals->dtype()) {
        case LLAISYS_DTYPE_F32:
            argmax_impl<float>(max_idx, max_val,
                               reinterpret_cast<const float*>(vals->data()), n);
            break;
        case LLAISYS_DTYPE_I32:
            argmax_impl<int32_t>(max_idx, max_val,
                                 reinterpret_cast<const int32_t*>(vals->data()), n);
            break;
        case LLAISYS_DTYPE_I8:
            argmax_impl<int8_t>(max_idx, max_val,
                                reinterpret_cast<const int8_t*>(vals->data()), n);
            break;
        case LLAISYS_DTYPE_I16:
            argmax_impl<int16_t>(max_idx, max_val,
                                 reinterpret_cast<const int16_t*>(vals->data()), n);
            break;
        case LLAISYS_DTYPE_I64:
            argmax_impl<int64_t>(max_idx, max_val,
                                 reinterpret_cast<const int64_t*>(vals->data()), n);
            break;
        case LLAISYS_DTYPE_U8:
            argmax_impl<uint8_t>(max_idx, max_val,
                                 reinterpret_cast<const uint8_t*>(vals->data()), n);
            break;
        case LLAISYS_DTYPE_U16:
            argmax_impl<uint16_t>(max_idx, max_val,
                                  reinterpret_cast<const uint16_t*>(vals->data()), n);
            break;
        case LLAISYS_DTYPE_U32:
            argmax_impl<uint32_t>(max_idx, max_val,
                                  reinterpret_cast<const uint32_t*>(vals->data()), n);
            break;
        case LLAISYS_DTYPE_U64:
            argmax_impl<uint64_t>(max_idx, max_val,
                                  reinterpret_cast<const uint64_t*>(vals->data()), n);
            break;
        case LLAISYS_DTYPE_F16:
            argmax_impl<fp16_t>(max_idx, max_val,
                                reinterpret_cast<const fp16_t*>(vals->data()), n);
            break;
        case LLAISYS_DTYPE_BF16:
            argmax_impl<bf16_t>(max_idx, max_val,
                                reinterpret_cast<const bf16_t*>(vals->data()), n);
            break;
        default:
            throw std::runtime_error("argmax: unsupported dtype");
    }
}

} //