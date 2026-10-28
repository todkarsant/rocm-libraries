// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "library_fixture.hpp"
#include "solution_entry.hpp"
#include <Tensile/Tensile.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <string>
#include <vector>

// Loads the JIT solution library written for this device and finds the
// pre-generated kernel's solutions, including both solutions of one published
// executable in one entry. Launches nothing.
namespace
{
    namespace fs = std::filesystem;
    namespace hj = hipblaslt_jit;
}

TEST_CASE("the loader reads the JIT solution library", "[jit-gpu]")
{
    const auto     target   = hipblaslt_jit_test::currentDeviceTarget();
    const auto     device   = target.device;
    const auto&    hardware = *target.hardware;
    const auto     key      = hipblaslt_jit_test::pregeneratedKey(target.properties);
    hj::JitLibrary library(fs::u8path(HIPBLASLT_JIT_LIBRARY));

    const auto prepared
        = hipblaslt_jit_test::prepareSolutions(fs::u8path(HIPBLASLT_JIT_DATA), target);
    const hipblaslt_jit_test::Prepared* pair = nullptr;
    bool                                loaded = false, paired = false;
    for(const auto& item : prepared)
    {
        if(item.name == "plain-pair")
            pair = &item;
        for(const auto k : item.ks)
        {
            const auto problem  = hipblaslt_jit_test::fp16Gemm(false, k);
            const auto expected = hj::supportingSolutions(
                item.source.solutions, problem, hardware, hipblaslt_jit_test::workspaceLimit);
            const auto           label = item.name + " K=" + std::to_string(k);
            std::vector<int32_t> indices;
            const auto status = library.lookup(key, device, problem, hardware, 4, {}, indices);
            JIT_REQUIRE("lookup " + label + ": " + status.message, status.ok());
            JIT_REQUIRE(label + " did not find one JIT solution index per published solution",
                        !expected.empty() && indices.size() == expected.size());
            for(size_t i = 0; i < indices.size(); ++i)
            {
                hj::Status why;
                const auto solution = library.solutionByIndex(device, hardware, indices[i], why);
                JIT_REQUIRE("Cannot load solution: " + why.message,
                            hj::isJitIndex(indices[i]) && solution != nullptr);
                JIT_REQUIRE("Found solution " + solution->solutionName + ", expected "
                                + expected[i]->solutionName,
                            solution->kernelName == expected[i]->kernelName
                                && solution->solutionName == expected[i]->solutionName);
                JIT_REQUIRE("The loaded solution has another index", solution->index == indices[i]);
                if(loaded)
                    continue;
                const auto view = library.resolve(device, indices[i], why);
                JIT_REQUIRE("resolve: " + why.message, view.master && view.adapter);
                const auto object = library.directory(key) / "slices" / hj::sliceKey(problem)
                                    / fs::u8path(std::string(solution->codeObjectFilename.load()));
                JIT_REQUIRE("Cannot load " + object.u8string(),
                            view.adapter->loadCodeObjectFile(object.string()) == hipSuccess);
                JIT_REQUIRE("Cannot resolve " + solution->kernelName,
                            view.adapter->initKernel(solution->kernelName) == hipSuccess);
                loaded = true;
            }
            if(indices.size() < 2)
                continue;

            // The solutions published from one executable share its entry.
            hj::Status why;
            const auto view = library.resolve(device, indices.front(), why);
            JIT_REQUIRE("resolve: " + why.message, view.master != nullptr);
            const auto& mapping = view.master->libraryMapping;
            const auto  prefix  = mapping.at(indices.front());
            for(const auto index : indices)
            {
                JIT_REQUIRE(label + " published its solutions into more than one entry",
                            mapping.count(index) && mapping.at(index) == prefix);
            }
            const auto file = library.directory(key) / "slices" / hj::sliceKey(problem)
                              / fs::u8path(prefix + ".dat");
            const auto entry = std::dynamic_pointer_cast<hj::GemmMaster>(
                TensileLite::LoadLibraryFile<TensileLite::ContractionProblemGemm>(file.string()));
            {
                INFO(("The entry " + file.u8string() + " does not hold every solution of "
                      + label));
                REQUIRE((entry && entry->solutions.size() == indices.size()));
                for(const auto index : indices)
                    REQUIRE((entry->solutions.count(index)));
            }
            paired = true;
        }
    }
    JIT_REQUIRE("plain-pair was not published as one entry, or nothing was loaded",
                pair && loaded && paired);

    std::vector<int32_t> transposed;
    const auto           status = library.lookup(
        key, device, hipblaslt_jit_test::fp16Gemm(true, 512), hardware, 4, {}, transposed);
    JIT_REQUIRE("A transposed A found a published solution", status.ok() && transposed.empty());
}
