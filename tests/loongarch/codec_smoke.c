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

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

size_t
streamvbyte_encode_0124(const uint32_t*, uint32_t, uint8_t*);
size_t
streamvbyte_decode_0124(const uint8_t*, uint32_t*, uint32_t);
size_t
streamvbyte_encode_0124_scalar(const uint32_t*, uint32_t, uint8_t*);
size_t
streamvbyte_decode_0124_scalar(const uint8_t*, uint32_t*, uint32_t);
size_t
read_ints(const uint8_t*, uint32_t*, int);
size_t
read_ints_single(const uint8_t*, uint32_t*, int);
size_t
masked_vbyte_read_loop(const uint8_t*, uint32_t*, int);
size_t
altmasked_vbyte_read_loop(const uint8_t*, uint32_t*, int);
size_t
masked_vbyte_read_loop_fromcompressedsize(const uint8_t*, uint32_t*, size_t);
size_t
altmasked_vbyte_read_loop_fromcompressedsize(const uint8_t*, uint32_t*, size_t);
void
knowhere_simd_pack_128_blocks(const uint32_t*, uint8_t*, size_t, uint32_t);
void
knowhere_simd_unpack_128_blocks(const uint8_t*, uint32_t*, size_t, uint32_t);
void
knowhere_simd_unpack_d1_128_blocks(const uint8_t*, uint32_t*, size_t, uint32_t, uint32_t);
void
knowhere_simd_integrate_doc_id_gaps(uint32_t*, size_t, uint32_t);

static uint32_t
next_random(uint32_t* state) {
    uint32_t value = *state;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    *state = value;
    return value;
}

static size_t
encode_varint(uint32_t value, uint8_t* out) {
    size_t size = 0;
    while (value >= 0x80) {
        out[size++] = (uint8_t)(value | 0x80);
        value >>= 7;
    }
    out[size++] = (uint8_t)value;
    return size;
}

static int
test_varint(void) {
    enum { count = 4096 };
    uint32_t input[count];
    uint32_t decoded[count + 16];
    uint8_t encoded[count * 5 + 64];
    uint32_t state = 0x87654321;
    size_t encoded_size = 0;

    for (size_t i = 0; i < count; ++i) {
        const uint32_t random = next_random(&state);
        switch (i % 7) {
            case 0:
                input[i] = 0;
                break;
            case 1:
                input[i] = random & 0x7f;
                break;
            case 2:
                input[i] = random & 0x3fff;
                break;
            case 3:
                input[i] = random & 0x1fffff;
                break;
            case 4:
                input[i] = random & 0xfffffff;
                break;
            default:
                input[i] = random;
                break;
        }
        encoded_size += encode_varint(input[i], encoded + encoded_size);
    }

#define CHECK_VARINT_DECODE(function, argument, expected)                                                 \
    do {                                                                                                  \
        memset(decoded, 0, sizeof(decoded));                                                              \
        const size_t result = function(encoded, decoded, argument);                                       \
        if (result != (expected) || memcmp(input, decoded, sizeof(input)) != 0) {                         \
            fprintf(stderr, #function " failed: result %zu, expected %zu\n", result, (size_t)(expected)); \
            return 1;                                                                                     \
        }                                                                                                 \
    } while (0)

    CHECK_VARINT_DECODE(read_ints, count, encoded_size);
    CHECK_VARINT_DECODE(read_ints_single, count, encoded_size);
    CHECK_VARINT_DECODE(masked_vbyte_read_loop, count, encoded_size);
    CHECK_VARINT_DECODE(altmasked_vbyte_read_loop, count, encoded_size);
    CHECK_VARINT_DECODE(masked_vbyte_read_loop_fromcompressedsize, encoded_size, count);
    CHECK_VARINT_DECODE(altmasked_vbyte_read_loop_fromcompressedsize, encoded_size, count);

#undef CHECK_VARINT_DECODE
    return 0;
}

static int
test_bitpacking(void) {
    enum { values_per_block = 128, max_packed_words = 128 };
    uint32_t input[values_per_block];
    uint32_t unpacked[values_per_block];
    uint32_t integrated[values_per_block];
    uint32_t expected_integrated[values_per_block];
    uint32_t expected_packed[max_packed_words];
    uint8_t packed[max_packed_words * sizeof(uint32_t)];
    uint32_t state = 0x31415926;

    for (uint32_t bits = 1; bits <= 32; ++bits) {
        const uint32_t mask = bits == 32 ? UINT32_MAX : (UINT32_C(1) << bits) - 1;
        const size_t packed_words = bits == 32 ? values_per_block : (size_t)bits * 4;
        memset(expected_packed, 0, sizeof(expected_packed));
        memset(packed, 0, sizeof(packed));
        for (size_t i = 0; i < values_per_block; ++i) {
            input[i] = next_random(&state) & mask;
        }

        if (bits == 32) {
            memcpy(expected_packed, input, sizeof(input));
        } else {
            for (size_t lane = 0; lane < 4; ++lane) {
                for (size_t value_index = 0; value_index < 32; ++value_index) {
                    const uint32_t value = input[value_index * 4 + lane];
                    const size_t bit_offset = value_index * bits;
                    const size_t word_index = bit_offset / 32;
                    const uint32_t shift = (uint32_t)(bit_offset % 32);
                    expected_packed[word_index * 4 + lane] |= value << shift;
                    if (shift + bits > 32) {
                        expected_packed[(word_index + 1) * 4 + lane] |= value >> (32 - shift);
                    }
                }
            }
        }

        knowhere_simd_pack_128_blocks(input, packed, 1, bits);
        if (memcmp(packed, expected_packed, packed_words * sizeof(uint32_t)) != 0) {
            fprintf(stderr, "bitpacking encode mismatch at width %u\n", bits);
            return 1;
        }

        memset(unpacked, 0, sizeof(unpacked));
        knowhere_simd_unpack_128_blocks(packed, unpacked, 1, bits);
        if (memcmp(input, unpacked, sizeof(input)) != 0) {
            fprintf(stderr, "bitpacking decode mismatch at width %u\n", bits);
            return 1;
        }

        if (bits < 32) {
            uint32_t previous = 17;
            for (size_t i = 0; i < values_per_block; ++i) {
                previous += input[i] + 1;
                expected_integrated[i] = previous;
            }
            knowhere_simd_unpack_d1_128_blocks(packed, integrated, 1, bits, 17);
            if (memcmp(integrated, expected_integrated, sizeof(integrated)) != 0) {
                fprintf(stderr, "bitpacking d1 mismatch at width %u\n", bits);
                return 1;
            }

            memcpy(integrated, input, sizeof(integrated));
            knowhere_simd_integrate_doc_id_gaps(integrated, values_per_block, 17);
            if (memcmp(integrated, expected_integrated, sizeof(integrated)) != 0) {
                fprintf(stderr, "bitpacking prefix mismatch at width %u\n", bits);
                return 1;
            }
        }
    }
    return 0;
}

int
main(void) {
    uint32_t state = 0x12345678;
    for (uint32_t count = 0; count <= 1024; ++count) {
        const size_t capacity = (size_t)count * 4 + (count + 3) / 4 + 32;
        uint32_t* input = calloc(count ? count : 1, sizeof(*input));
        uint32_t* decoded_lsx = calloc(count ? count : 1, sizeof(*decoded_lsx));
        uint32_t* decoded_scalar = calloc(count ? count : 1, sizeof(*decoded_scalar));
        uint8_t* encoded_lsx = calloc(capacity, 1);
        uint8_t* encoded_scalar = calloc(capacity, 1);
        if (!input || !decoded_lsx || !decoded_scalar || !encoded_lsx || !encoded_scalar) {
            return 2;
        }

        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t selector = i & 7;
            const uint32_t random = next_random(&state);
            input[i] = selector == 0   ? 0
                       : selector == 1 ? random & 0xff
                       : selector == 2 ? random & 0xffff
                       : selector == 3 ? random
                                       : random % 1024;
        }

        const size_t lsx_size = streamvbyte_encode_0124(input, count, encoded_lsx);
        const size_t scalar_size = streamvbyte_encode_0124_scalar(input, count, encoded_scalar);
        if (lsx_size != scalar_size || memcmp(encoded_lsx, encoded_scalar, scalar_size) != 0) {
            fprintf(stderr, "encode mismatch at count %u: LSX %zu, scalar %zu\n", count, lsx_size, scalar_size);
            return 1;
        }

        const size_t lsx_read = streamvbyte_decode_0124(encoded_scalar, decoded_lsx, count);
        const size_t scalar_read = streamvbyte_decode_0124_scalar(encoded_scalar, decoded_scalar, count);
        if (lsx_read != scalar_read || lsx_read != scalar_size ||
            memcmp(input, decoded_lsx, (size_t)count * sizeof(*input)) != 0 ||
            memcmp(input, decoded_scalar, (size_t)count * sizeof(*input)) != 0) {
            fprintf(stderr, "decode mismatch at count %u: LSX %zu, scalar %zu, encoded %zu\n", count, lsx_read,
                    scalar_read, scalar_size);
            return 1;
        }

        free(input);
        free(decoded_lsx);
        free(decoded_scalar);
        free(encoded_lsx);
        free(encoded_scalar);
    }

    if (test_varint() != 0) {
        return 1;
    }
    if (test_bitpacking() != 0) {
        return 1;
    }

    puts("Sparse codec LSX smoke tests passed");
    return 0;
}
