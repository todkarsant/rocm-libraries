// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

#include <tensilelitehost/export.h>

namespace TensileLite
{
    // The static catalogs that SolutionState.cpp accepts when it rebuilds a
    // record from state. Exported for tensilelite-tests; this header is not
    // installed.
    TENSILELITEHOST_EXPORT std::vector<std::string> supportedBuildChipIds();
    TENSILELITEHOST_EXPORT std::vector<std::string> supportedDataTypeNames();
}
