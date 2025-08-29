#include "op.hpp"

#include <cstring>
#include <stdexcept>

namespace llaisys::ops {

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // basic checks
    if (index->dtype() != LLAISYS_DTYPE_I64) {
        throw std::runtime_error("embedding: index must be int64");
    }
    if (index->ndim() != 1) {
        throw std::runtime_error("embedding: index must be 1-D");
    }
    if (weight->ndim() != 2) {
        throw std::runtime_error("embedding: weight must be 2-D");
    }
    if (out->ndim() != 2) {
        throw std::runtime_error("embedding: out must be 2-D");
    }

    const auto &idx_shape = index->shape();
    const auto &w_shape   = weight->shape();
    const auto &out_shape = out->shape();

    if (out_shape[0] != idx_shape[0] || out_shape[1] != w_shape[1]) {
        throw std::runtime_error("embedding: output shape must be (len(index), weight.shape[1])");
    }

    size_t rows = idx_shape[0];
    size_t cols = w_shape[1];
    size_t elem_bytes = weight->elementSize();

    // strides are in elements; convert to bytes when computing pointer offsets
    const auto &w_strides = weight->strides();
    const auto &out_strides = out->strides();
    ptrdiff_t w_row_stride_e = w_strides[0];
    ptrdiff_t out_row_stride_e = out_strides[0];

    const int64_t *idx_ptr = reinterpret_cast<const int64_t *>(index->data());
    const std::byte *w_base = reinterpret_cast<const std::byte *>(weight->data());
    std::byte *out_base = reinterpret_cast<std::byte *>(out->data());

    for (size_t i = 0; i < rows; ++i) {
        int64_t src_row = idx_ptr[i];
        if (src_row < 0 || static_cast<size_t>(src_row) >= w_shape[0]) {
            throw std::out_of_range("embedding: index out of range");
        }

        const std::byte *src = w_base + (static_cast<size_t>(src_row) * static_cast<size_t>(w_row_stride_e) * elem_bytes);
        std::byte *dst = out_base + (i * static_cast<size_t>(out_row_stride_e) * elem_bytes);

        // copy a contiguous row of 'cols' elements (cols * element_size bytes)
        std::memcpy(dst, src, cols * elem_bytes);
    }
}

} // namespace llaisys::ops