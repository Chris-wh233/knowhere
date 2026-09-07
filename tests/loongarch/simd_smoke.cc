// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
// an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <string_view>
#include <vector>

#include "index/sparse/sindi_simd.h"
#include "src/simd/distances_lasx.h"
#include "src/simd/distances_lsx.h"

namespace kh = faiss::cppcontrib::knowhere;

using BinaryFn = float (*)(const float*, const float*, size_t);
using NormFn = float (*)(const float*, size_t);
using NyFn = void (*)(float*, const float*, const float*, size_t, size_t);
using TransposedFn = void (*)(float*, const float*, const float*, const float*, size_t, size_t, size_t);
using NearestFn = size_t (*)(float*, const float*, const float*, size_t, size_t);
using TransposedNearestFn = size_t (*)(float*, const float*, const float*, const float*, size_t, size_t, size_t);
using MaddFn = void (*)(size_t, const float*, float, const float*, float*);
using MaddArgminFn = int (*)(size_t, const float*, float, const float*, float*);
using BatchFn = void (*)(const float*, const float*, const float*, const float*, const float*, size_t, float&, float&,
                         float&, float&);

struct Backend {
    const char* name;
    BinaryFn ip;
    BinaryFn l2;
    BinaryFn l1;
    BinaryFn linf;
    NormFn norm;
    NyFn l2_ny;
    NyFn ip_ny;
    TransposedFn transposed;
    NearestFn nearest;
    TransposedNearestFn transposed_nearest;
    MaddFn madd;
    MaddArgminFn madd_argmin;
    BatchFn ip_batch;
    BatchFn l2_batch;
};

float
ip_ref(const float* x, const float* y, size_t d) {
    float result = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        result += x[i] * y[i];
    }
    return result;
}

float
l2_ref(const float* x, const float* y, size_t d) {
    float result = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        const float diff = x[i] - y[i];
        result += diff * diff;
    }
    return result;
}

float
l1_ref(const float* x, const float* y, size_t d) {
    float result = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        result += std::fabs(x[i] - y[i]);
    }
    return result;
}

float
linf_ref(const float* x, const float* y, size_t d) {
    float result = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        result = std::fmax(result, std::fabs(x[i] - y[i]));
    }
    return result;
}

bool
close(float actual, float expected) {
    return std::fabs(actual - expected) <= 2e-5f * (1.0f + std::fabs(expected));
}

void
require_close(const Backend& backend, const char* operation, float actual, float expected) {
    if (!close(actual, expected)) {
        std::cerr << backend.name << ' ' << operation << ": " << actual << " != " << expected << '\n';
        std::exit(1);
    }
}

const Backend&
select_backend(std::string_view name) {
    static const Backend lsx = {
        "LSX",
        kh::fvec_inner_product_lsx,
        kh::fvec_L2sqr_lsx,
        kh::fvec_L1_lsx,
        kh::fvec_Linf_lsx,
        kh::fvec_norm_L2sqr_lsx,
        kh::fvec_L2sqr_ny_lsx,
        kh::fvec_inner_products_ny_lsx,
        kh::fvec_L2sqr_ny_transposed_lsx,
        kh::fvec_L2sqr_ny_nearest_lsx,
        kh::fvec_L2sqr_ny_nearest_y_transposed_lsx,
        kh::fvec_madd_lsx,
        kh::fvec_madd_and_argmin_lsx,
        kh::fvec_inner_product_batch_4_lsx,
        kh::fvec_L2sqr_batch_4_lsx,
    };
    static const Backend lasx = {
        "LASX",
        kh::fvec_inner_product_lasx,
        kh::fvec_L2sqr_lasx,
        kh::fvec_L1_lasx,
        kh::fvec_Linf_lasx,
        kh::fvec_norm_L2sqr_lasx,
        kh::fvec_L2sqr_ny_lasx,
        kh::fvec_inner_products_ny_lasx,
        kh::fvec_L2sqr_ny_transposed_lasx,
        kh::fvec_L2sqr_ny_nearest_lasx,
        kh::fvec_L2sqr_ny_nearest_y_transposed_lasx,
        kh::fvec_madd_lasx,
        kh::fvec_madd_and_argmin_lasx,
        kh::fvec_inner_product_batch_4_lasx,
        kh::fvec_L2sqr_batch_4_lasx,
    };

    if (name == "lsx") {
        return lsx;
    }
    if (name == "lasx") {
        return lasx;
    }
    std::cerr << "usage: simd_smoke <lsx|lasx>\n";
    std::exit(2);
}

void
test_dimension(const Backend& backend, size_t d, std::mt19937& rng) {
    constexpr size_t ny = 17;
    std::uniform_real_distribution<float> distribution(-10.0f, 10.0f);
    std::vector<float> x(d);
    std::vector<float> y(ny * d);
    for (float& value : x) {
        value = distribution(rng);
    }
    for (float& value : y) {
        value = distribution(rng);
    }

    require_close(backend, "ip", backend.ip(x.data(), y.data(), d), ip_ref(x.data(), y.data(), d));
    require_close(backend, "l2", backend.l2(x.data(), y.data(), d), l2_ref(x.data(), y.data(), d));
    require_close(backend, "l1", backend.l1(x.data(), y.data(), d), l1_ref(x.data(), y.data(), d));
    require_close(backend, "linf", backend.linf(x.data(), y.data(), d), linf_ref(x.data(), y.data(), d));
    require_close(backend, "norm", backend.norm(x.data(), d), ip_ref(x.data(), x.data(), d));
    if (d == 0) {
        return;
    }

    std::vector<float> distances(ny), products(ny);
    backend.l2_ny(distances.data(), x.data(), y.data(), d, ny);
    backend.ip_ny(products.data(), x.data(), y.data(), d, ny);
    for (size_t i = 0; i < ny; ++i) {
        require_close(backend, "l2_ny", distances[i], l2_ref(x.data(), y.data() + i * d, d));
        require_close(backend, "ip_ny", products[i], ip_ref(x.data(), y.data() + i * d, d));
    }

    constexpr size_t offset = 20;
    std::vector<float> transposed(offset * d, 0.0f), y_sqlen(ny);
    for (size_t i = 0; i < ny; ++i) {
        y_sqlen[i] = ip_ref(y.data() + i * d, y.data() + i * d, d);
        for (size_t j = 0; j < d; ++j) {
            transposed[i + j * offset] = y[i * d + j];
        }
    }
    backend.transposed(distances.data(), x.data(), transposed.data(), y_sqlen.data(), d, offset, ny);
    for (size_t i = 0; i < ny; ++i) {
        require_close(backend, "transposed", distances[i], l2_ref(x.data(), y.data() + i * d, d));
    }

    std::vector<float> nearest_buffer(ny);
    const size_t expected_nearest =
        static_cast<size_t>(std::min_element(distances.begin(), distances.end()) - distances.begin());
    if (backend.nearest(nearest_buffer.data(), x.data(), y.data(), d, ny) != expected_nearest ||
        backend.transposed_nearest(nearest_buffer.data(), x.data(), transposed.data(), y_sqlen.data(), d, offset, ny) !=
            expected_nearest) {
        std::cerr << backend.name << " nearest mismatch\n";
        std::exit(1);
    }

    std::vector<float> madd(d), madd_argmin(d);
    backend.madd(d, x.data(), 1.25f, y.data(), madd.data());
    const int argmin = backend.madd_argmin(d, x.data(), 1.25f, y.data(), madd_argmin.data());
    int expected_argmin = -1;
    float minimum = std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < d; ++i) {
        const float expected = x[i] + 1.25f * y[i];
        require_close(backend, "madd", madd[i], expected);
        require_close(backend, "madd_argmin", madd_argmin[i], expected);
        if (expected < minimum) {
            minimum = expected;
            expected_argmin = static_cast<int>(i);
        }
    }
    if (argmin != expected_argmin) {
        std::cerr << backend.name << " argmin mismatch\n";
        std::exit(1);
    }

    float ip[4], l2[4];
    backend.ip_batch(x.data(), y.data(), y.data() + d, y.data() + 2 * d, y.data() + 3 * d, d, ip[0], ip[1], ip[2],
                     ip[3]);
    backend.l2_batch(x.data(), y.data(), y.data() + d, y.data() + 2 * d, y.data() + 3 * d, d, l2[0], l2[1], l2[2],
                     l2[3]);
    for (size_t i = 0; i < 4; ++i) {
        require_close(backend, "ip_batch", ip[i], ip_ref(x.data(), y.data() + i * d, d));
        require_close(backend, "l2_batch", l2[i], l2_ref(x.data(), y.data() + i * d, d));
    }
}

void
test_datatypes_lsx() {
    const Backend& backend = select_backend("lsx");
    for (size_t d : {size_t{0}, size_t{1}, size_t{7}, size_t{8}, size_t{15}, size_t{16}, size_t{31}, size_t{65}}) {
        std::vector<knowhere::fp16> fp16_x(d), fp16_y(d);
        std::vector<knowhere::bf16> bf16_x(d), bf16_y(d);
        std::vector<int8_t> int8_x(d), int8_y(d);
        float fp16_ip = 0.0f;
        float fp16_l2 = 0.0f;
        float fp16_norm = 0.0f;
        float bf16_ip = 0.0f;
        float bf16_l2 = 0.0f;
        float bf16_norm = 0.0f;
        float int8_ip = 0.0f;
        float int8_l2 = 0.0f;
        float int8_norm = 0.0f;
        int32_t ivec_ip = 0;
        int32_t ivec_l2 = 0;

        for (size_t i = 0; i < d; ++i) {
            fp16_x[i] = static_cast<float>(static_cast<int>(i % 19) - 9) / 8.0f;
            fp16_y[i] = static_cast<float>(static_cast<int>((i * 7) % 23) - 11) / 4.0f;
            bf16_x[i] = static_cast<float>(static_cast<int>((i * 3) % 17) - 8) / 2.0f;
            bf16_y[i] = static_cast<float>(static_cast<int>((i * 5) % 29) - 14) / 4.0f;
            int8_x[i] = static_cast<int8_t>(static_cast<int>((i * 11) % 251) - 125);
            int8_y[i] = static_cast<int8_t>(static_cast<int>((i * 13) % 247) - 123);

            const float fp16_x_value = static_cast<float>(fp16_x[i]);
            const float fp16_y_value = static_cast<float>(fp16_y[i]);
            const float fp16_diff = fp16_x_value - fp16_y_value;
            fp16_ip += fp16_x_value * fp16_y_value;
            fp16_l2 += fp16_diff * fp16_diff;
            fp16_norm += fp16_x_value * fp16_x_value;

            const float bf16_x_value = static_cast<float>(bf16_x[i]);
            const float bf16_y_value = static_cast<float>(bf16_y[i]);
            const float bf16_diff = bf16_x_value - bf16_y_value;
            bf16_ip += bf16_x_value * bf16_y_value;
            bf16_l2 += bf16_diff * bf16_diff;
            bf16_norm += bf16_x_value * bf16_x_value;

            const int32_t x_value = int8_x[i];
            const int32_t y_value = int8_y[i];
            const int32_t int8_diff = x_value - y_value;
            ivec_ip += x_value * y_value;
            ivec_l2 += int8_diff * int8_diff;
            int8_ip += static_cast<float>(x_value * y_value);
            int8_l2 += static_cast<float>(int8_diff * int8_diff);
            int8_norm += static_cast<float>(x_value * x_value);
        }

        require_close(backend, "fp16_ip", kh::fp16_vec_inner_product_lsx(fp16_x.data(), fp16_y.data(), d), fp16_ip);
        require_close(backend, "fp16_l2", kh::fp16_vec_L2sqr_lsx(fp16_x.data(), fp16_y.data(), d), fp16_l2);
        require_close(backend, "fp16_norm", kh::fp16_vec_norm_L2sqr_lsx(fp16_x.data(), d), fp16_norm);
        require_close(backend, "bf16_ip", kh::bf16_vec_inner_product_lsx(bf16_x.data(), bf16_y.data(), d), bf16_ip);
        require_close(backend, "bf16_l2", kh::bf16_vec_L2sqr_lsx(bf16_x.data(), bf16_y.data(), d), bf16_l2);
        require_close(backend, "bf16_norm", kh::bf16_vec_norm_L2sqr_lsx(bf16_x.data(), d), bf16_norm);
        require_close(backend, "int8_ip", kh::int8_vec_inner_product_lsx(int8_x.data(), int8_y.data(), d), int8_ip);
        require_close(backend, "int8_l2", kh::int8_vec_L2sqr_lsx(int8_x.data(), int8_y.data(), d), int8_l2);
        require_close(backend, "int8_norm", kh::int8_vec_norm_L2sqr_lsx(int8_x.data(), d), int8_norm);

        float results[4];
        kh::fp16_vec_inner_product_batch_4_lsx(fp16_x.data(), fp16_y.data(), fp16_x.data(), fp16_y.data(),
                                               fp16_x.data(), d, results[0], results[1], results[2], results[3]);
        for (size_t i = 0; i < 4; ++i) {
            require_close(backend, "fp16_ip_batch", results[i], i % 2 == 0 ? fp16_ip : fp16_norm);
        }
        kh::fp16_vec_L2sqr_batch_4_lsx(fp16_x.data(), fp16_y.data(), fp16_x.data(), fp16_y.data(), fp16_x.data(), d,
                                       results[0], results[1], results[2], results[3]);
        for (size_t i = 0; i < 4; ++i) {
            require_close(backend, "fp16_l2_batch", results[i], i % 2 == 0 ? fp16_l2 : 0.0f);
        }

        kh::bf16_vec_inner_product_batch_4_lsx(bf16_x.data(), bf16_y.data(), bf16_x.data(), bf16_y.data(),
                                               bf16_x.data(), d, results[0], results[1], results[2], results[3]);
        for (size_t i = 0; i < 4; ++i) {
            require_close(backend, "bf16_ip_batch", results[i], i % 2 == 0 ? bf16_ip : bf16_norm);
        }
        kh::bf16_vec_L2sqr_batch_4_lsx(bf16_x.data(), bf16_y.data(), bf16_x.data(), bf16_y.data(), bf16_x.data(), d,
                                       results[0], results[1], results[2], results[3]);
        for (size_t i = 0; i < 4; ++i) {
            require_close(backend, "bf16_l2_batch", results[i], i % 2 == 0 ? bf16_l2 : 0.0f);
        }

        kh::int8_vec_inner_product_batch_4_lsx(int8_x.data(), int8_y.data(), int8_x.data(), int8_y.data(),
                                               int8_x.data(), d, results[0], results[1], results[2], results[3]);
        for (size_t i = 0; i < 4; ++i) {
            require_close(backend, "int8_ip_batch", results[i], i % 2 == 0 ? int8_ip : int8_norm);
        }
        kh::int8_vec_L2sqr_batch_4_lsx(int8_x.data(), int8_y.data(), int8_x.data(), int8_y.data(), int8_x.data(), d,
                                       results[0], results[1], results[2], results[3]);
        for (size_t i = 0; i < 4; ++i) {
            require_close(backend, "int8_l2_batch", results[i], i % 2 == 0 ? int8_l2 : 0.0f);
        }

        if (kh::ivec_inner_product_lsx(int8_x.data(), int8_y.data(), d) != ivec_ip ||
            kh::ivec_L2sqr_lsx(int8_x.data(), int8_y.data(), d) != ivec_l2) {
            std::cerr << "LSX integer distance mismatch at dimension " << d << '\n';
            std::exit(1);
        }
    }
}

void
test_sindi_lsx() {
    constexpr int32_t count = 17;
    constexpr size_t output_size = 67;
    constexpr float qval = 1.25f;
    std::vector<knowhere::fp16> fp16_values(count);
    std::vector<uint16_t> tf_values(count), ids(count);
    std::vector<float> row_sums(output_size), expected(output_size), actual(output_size);
    for (size_t i = 0; i < output_size; ++i) {
        row_sums[i] = static_cast<float>(i % 11 + 1);
    }
    for (int32_t i = 0; i < count; ++i) {
        fp16_values[i] = static_cast<float>(i + 1) / 8.0f;
        tf_values[i] = static_cast<uint16_t>(i + 1);
        ids[i] = static_cast<uint16_t>((i * 7 + 3) % output_size);
        expected[ids[i]] += qval * static_cast<float>(fp16_values[i]);
    }

    const float ip_max = knowhere::sparse::inverted::sindi::ip_accumulate_lsx_fp16(qval, fp16_values.data(), ids.data(),
                                                                                   count, actual.data());
    float expected_max = 0.0f;
    for (size_t i = 0; i < output_size; ++i) {
        require_close(select_backend("lsx"), "sindi_ip", actual[i], expected[i]);
        expected_max = std::fmax(expected_max, expected[i]);
    }
    require_close(select_backend("lsx"), "sindi_ip_max", ip_max, expected_max);

    std::fill(expected.begin(), expected.end(), 0.0f);
    std::fill(actual.begin(), actual.end(), 0.0f);
    constexpr float k1 = 1.2f;
    constexpr float b = 0.75f;
    constexpr float avgdl = 8.0f;
    for (int32_t i = 0; i < count; ++i) {
        const float tf = static_cast<float>(tf_values[i]);
        const float score = qval * (k1 + 1.0f) * tf / (tf + k1 * (1.0f - b) + k1 * b / avgdl * row_sums[ids[i]]);
        expected[ids[i]] += score;
    }
    const float bm25_max = knowhere::sparse::inverted::sindi::bm25_accumulate_lsx_u16(
        qval, tf_values.data(), ids.data(), count, actual.data(), k1, b, avgdl, row_sums.data());
    expected_max = 0.0f;
    for (size_t i = 0; i < output_size; ++i) {
        require_close(select_backend("lsx"), "sindi_bm25", actual[i], expected[i]);
        expected_max = std::fmax(expected_max, expected[i]);
    }
    require_close(select_backend("lsx"), "sindi_bm25_max", bm25_max, expected_max);

    const std::vector<float> scores = {1.0f, 5.0f, 3.0f, -1.0f, 8.0f, 2.0f};
    knowhere::ResultMinHeap<float, uint32_t> heap(3);
    float threshold = 0.0f;
    knowhere::sparse::inverted::sindi::batch_insert_lsx(scores.data(), 10, scores.size(), heap, threshold, {});
    heap.Finalize();
    const std::vector<std::pair<float, uint32_t>> expected_results = {{8.0f, 14}, {5.0f, 11}, {3.0f, 12}};
    if (heap.Results() != expected_results || threshold != 3.0f) {
        std::cerr << "LSX SINDI batch insert mismatch\n";
        std::exit(1);
    }
}

int
main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: simd_smoke <lsx|lasx>\n";
        return 2;
    }
    const Backend& backend = select_backend(argv[1]);
    std::mt19937 rng(42);
    for (size_t d : {size_t{0}, size_t{1}, size_t{3}, size_t{4}, size_t{7}, size_t{8}, size_t{13}, size_t{31},
                     size_t{64}, size_t{129}}) {
        test_dimension(backend, d, rng);
    }
    if (std::string_view(argv[1]) == "lsx") {
        test_datatypes_lsx();
        test_sindi_lsx();
    }
    std::cout << backend.name << " smoke test passed\n";
}
