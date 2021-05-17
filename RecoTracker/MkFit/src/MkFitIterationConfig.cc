#include "RecoTracker/MkFit/interface/MkFitIterationConfig.h"

#include "mkFit/IterationConfig.h"

MkFitIterationConfig::MkFitIterationConfig(std::unique_ptr<mkfit::IterationsInfo> info,
                                           const mkfit::IterationConfig* config)
    : iterationsInfo_{std::move(info)}, config_{config} {}

MkFitIterationConfig::~MkFitIterationConfig() = default;
