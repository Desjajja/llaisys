#include "op.hpp"

#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>

namespace llaisys::ops {

// convert half (IEEE 754 binary16 stored in uint16_t) to float
static inline float half_to_float(uint16_t h) {
    const uint32_t sign = (h >> 15) & 0x1;
    const uint32_t exp  = (h >> 10) & 0x1F;
    const uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            return sign ? -0.0f : 0.0f;
        } else {
            // subnormal
            float m = mant / 1024.0f; // mantissa / 2^10
            float val = std::ldexp(m, -14); // m * 2^-14
            return sign ? -val : val;
        }
    } else if (exp == 31) {
        if (mant == 0) {
            return sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        } else {
            return std::numeric_limits<float>::quiet_NaN();
        }
    } else {
        float m = 1.0f + (mant / 1024.0f);
        float val = std::ldexp(m, static_cast<int>(exp) - 15);
        return sign ? -val : val;
    }
}

// convert float to half (IEEE 754 binary16 packed into uint16_t)
// simple implementation with rounding-to-nearest-even approximation
static inline uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = ((x >> 23) & 0xFF) - 127;
    uint32_t mant = x & 0x7FFFFFu;

    if (((x >> 23) & 0xFF) == 0xFF) { // NaN or Inf
        if (mant == 0) { // Inf
            return static_cast<uint16_t>(sign | 0x7C00u);
        } else { // NaN
            return static_cast<uint16_t>(sign | 0x7E00u);
        }
    }

    int32_t newexp = exp + 15;
    if (newexp >= 0x1F) { // overflow -> Inf
        return static_cast<uint16_t>(sign | 0x7C00u);
    } else if (newexp <= 0) {
        // subnormal or underflow to zero
        if (newexp < -10) {
            return static_cast<uint16_t>(sign); // zero
        }
        // create subnormal half
        mant = mant | 0x800000u; // add implicit 1
        int shift = 14 - newexp;
        uint32_t half_mant = mant >> (shift + 13); // mant >> (shift + (23-10))
        // rounding: check bit just below cutoff
        uint32_t rem = (mant >> (shift + 12)) & 1u;
        half_mant += rem;
        return static_cast<uint16_t>(sign | half_mant);
    } else {
        uint16_t half = static_cast<uint16_t>(sign | (static_cast<uint16_t>(newexp) << 10) | static_cast<uint16_t>(mant >> 13));
        // rounding: check bit 12 of mant (the highest dropped bit)
        if (mant & 0x00001000u) {
            half++; // simple round
        }
        return half;
    }
}

// convert bfloat16 (stored as uint16_t) to float32
static inline float bf16_to_float(uint16_t b) {
    uint32_t x = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &x, sizeof(f));
    return f;
}

// convert float32 to bfloat16 (uint16_t) with simple round-to-nearest
static inline uint16_t float_to_bf16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    // round to nearest even: add 0x8000 then shift
    uint32_t rounding = 0x8000u;
    uint16_t b = static_cast<uint16_t>((x + rounding) >> 16);
    return b;
}

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    size_t n = vals->numel();
    if (n == 0) {
        throw std::runtime_error("argmax: empty tensor");
    }

    switch (vals->dtype()) {
        case LLAISYS_DTYPE_F32: {
            auto vals_data = reinterpret_cast<const float*>(vals->data());
            float max_value = vals_data[0];
            size_t max_index = 0;
            for (size_t i = 1; i < n; ++i) {
                if (vals_data[i] > max_value) {
                    max_value = vals_data[i];
                    max_index = i;
                }
            }
            *reinterpret_cast<int64_t*>(max_idx->data()) = static_cast<int64_t>(max_index);
            *reinterpret_cast<float*>(max_val->data()) = max_value;
            break;
        }

        case LLAISYS_DTYPE_I32: {
            auto vals_data = reinterpret_cast<const int32_t*>(vals->data());
            int32_t max_value = vals_data[0];
            size_t max_index = 0;
            for (size_t i = 1; i < n; ++i) {
                if (vals_data[i] > max_value) {
                    max_value = vals_data[i];
                    max_index = i;
                }
            }
            *reinterpret_cast<int64_t*>(max_idx->data()) = static_cast<int64_t>(max_index);
            *reinterpret_cast<int32_t*>(max_val->data()) = max_value;
            break;
        }

        case LLAISYS_DTYPE_F16: {
            auto vals_data = reinterpret_cast<const uint16_t*>(vals->data());
            float max_value_f = half_to_float(vals_data[0]);
            size_t max_index = 0;
            for (size_t i = 1; i < n; ++i) {
                float v = half_to_float(vals_data[i]);
                if (v > max_value_f) {
                    max_value_f = v;
                    max_index = i;
                }
            }
            *reinterpret_cast<int64_t*>(max_idx->data()) = static_cast<int64_t>(max_index);
            uint16_t out = float_to_half(max_value_f);
            *reinterpret_cast<uint16_t*>(max_val->data()) = out;
            break;
        }

        case LLAISYS_DTYPE_BF16: {
            auto vals_data = reinterpret_cast<const uint16_t*>(vals->data());
            float max_value_f = bf16_to_float(vals_data[0]);
            size_t max_index = 0;
            for (size_t i = 1; i < n; ++i) {
                float v = bf16_to_float(vals_data[i]);
                if (v > max_value_f) {
                    max_value_f = v;
                    max_index = i;
                }
            }
            *reinterpret_cast<int64_t*>(max_idx->data()) = static_cast<int64_t>(max_index);
            uint16_t out = float_to_bf16(max_value_f);
            *reinterpret_cast<uint16_t*>(max_val->data()) = out;
            break;
        }

        default:
            throw std::runtime_error("argmax: unsupported dtype");
    }
}

}