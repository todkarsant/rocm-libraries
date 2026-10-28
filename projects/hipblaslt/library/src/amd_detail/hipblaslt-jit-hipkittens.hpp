// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-code-object.hpp"
#include <Tensile/ContractionSolution.hpp>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace hipblaslt_jit
{
    class Jit;
}

// Not installed. A backend that compiles HipKittens kernel templates with comgr,
// in process. Each kernel variant serves a narrow set of problems; others are
// NOT_SUPPORTED. The HipKittens headers the templates include are installed
// with the library under hipblaslt/hipkittens/<commit>.
namespace hipblaslt_ext::experimental::jit::hipkittens
{
    // Reads and verifies the headers in HIPBLASLT_JIT_HIPKITTENS_PATH or, when
    // that is unset or empty, hipblaslt/hipkittens/<commit> next to libhipblaslt.
    // When they are missing or do not match their manifest, fails with
    // "JIT backend HipKittens not available: ...".
    HIPBLASLT_EXPORT hipblasStatus_t createBackend(std::shared_ptr<const hipblaslt_jit::Jit>& jit,
                                                   Diagnostics& diagnostics);

    namespace detail
    {
        // One installed header: its path in the header directory, its size and
        // the SHA-256 the runtime checks. The bytes the compiler includes are an
        // Artifact, read from that path.
        struct HeaderFile
        {
            std::filesystem::path path;
            uint64_t              size = 0;
            std::string           sha256;
        };

        // One HipKittens kernel: the HIP main kernel, the flags it compiles
        // with, and the TN GEMM it implements for one processor, with A, B, C,
        // D and E of one data type.
        struct Kernel : hipblaslt_jit::Artifact
        {
            hipblaslt_jit::Processor processor;
            std::string              kernelName;
            rocisa::DataType         type;
            std::vector<std::string> hipFlags;

            // The solution an executable built from this kernel runs. Its
            // WorkgroupMappingXCCCheck predicate reads the current HIP device.
            HIPBLASLT_EXPORT std::shared_ptr<TensileLite::ContractionSolution> solution() const;
        };
        HIPBLASLT_EXPORT const std::vector<Kernel>& kernels();

        // Generated at configure time by cmake/hipblaslt-jit-hipkittens.cmake.
        // manifest is the text of manifest.txt, installed next to the headers,
        // and source the HIP template every kernel compiles.
        struct Resources
        {
            std::vector<uint8_t>    manifest;
            std::vector<HeaderFile> headers;
            std::vector<uint8_t>    source;
        };
        HIPBLASLT_EXPORT const Resources& resources();
    }
}
