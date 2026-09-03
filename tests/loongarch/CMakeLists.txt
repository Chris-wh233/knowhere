# Copyright (C) 2019-2023 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

cmake_minimum_required(VERSION 3.20)
project(knowhere_loongarch_simd_smoke LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

get_filename_component(KNOWHERE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

add_library(simd_lsx OBJECT "${KNOWHERE_SOURCE_DIR}/src/simd/distances_lsx.cc")
target_include_directories(simd_lsx PRIVATE "${KNOWHERE_SOURCE_DIR}/include")
target_compile_options(simd_lsx PRIVATE -O3 -Wall -Wextra -Werror -mlsx -mno-lasx)

add_library(simd_lasx OBJECT "${KNOWHERE_SOURCE_DIR}/src/simd/distances_lasx.cc")
target_compile_options(simd_lasx PRIVATE -O3 -Wall -Wextra -Werror -mlasx)

add_library(sindi_simd_lsx OBJECT "${KNOWHERE_SOURCE_DIR}/src/index/sparse/sindi_simd_lsx.cc")
target_include_directories(
  sindi_simd_lsx PRIVATE "${KNOWHERE_SOURCE_DIR}/include"
                         "${KNOWHERE_SOURCE_DIR}/src")
target_compile_options(sindi_simd_lsx PRIVATE -O3 -Wall -Wextra -Werror -mlsx -mno-lasx)

add_executable(
  simd_smoke simd_smoke.cc $<TARGET_OBJECTS:simd_lsx>
             $<TARGET_OBJECTS:simd_lasx> $<TARGET_OBJECTS:sindi_simd_lsx>)
target_include_directories(
  simd_smoke PRIVATE "${KNOWHERE_SOURCE_DIR}" "${KNOWHERE_SOURCE_DIR}/include"
                     "${KNOWHERE_SOURCE_DIR}/src")
target_compile_options(simd_smoke PRIVATE -O3 -Wall -Wextra -Werror -mlsx -mno-lasx)

set(STREAMVBYTE_SOURCES
    "${KNOWHERE_SOURCE_DIR}/src/index/sparse/codec/streamvbyte_0124_decode.c"
    "${KNOWHERE_SOURCE_DIR}/src/index/sparse/codec/streamvbyte_0124_encode.c")

add_library(
  codec_lsx OBJECT
  ${STREAMVBYTE_SOURCES}
  "${KNOWHERE_SOURCE_DIR}/src/index/sparse/codec/simd_bitpacking_kernel.c"
  "${KNOWHERE_SOURCE_DIR}/src/index/sparse/codec/varintdecode.c")
target_include_directories(codec_lsx PRIVATE "${KNOWHERE_SOURCE_DIR}/src")
target_compile_options(codec_lsx PRIVATE -O3 -Wall -Wextra -Werror -mlsx -mno-lasx)

add_library(streamvbyte_scalar OBJECT ${STREAMVBYTE_SOURCES})
target_compile_options(streamvbyte_scalar PRIVATE -O3 -Wall -Wextra -Werror -mno-lsx -mno-lasx)
target_compile_definitions(
  streamvbyte_scalar
  PRIVATE streamvbyte_decode_0124=streamvbyte_decode_0124_scalar
          streamvbyte_encode_0124=streamvbyte_encode_0124_scalar
          streamvbyte_validate_stream_0124=streamvbyte_validate_stream_0124_scalar)

add_executable(
  codec_smoke codec_smoke.c $<TARGET_OBJECTS:codec_lsx>
              $<TARGET_OBJECTS:streamvbyte_scalar>)
target_compile_options(codec_smoke PRIVATE -O3 -Wall -Wextra -Werror -mlsx -mno-lasx)
