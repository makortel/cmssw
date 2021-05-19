#include "RecoTracker/MkFit/interface/MkFitIterationConfig.h"

#include "mkFit/IterationConfig.h"

MkFitIterationConfig::MkFitIterationConfig(std::unique_ptr<mkfit::IterationConfig> config)
    : config_{std::move(config)} {}

MkFitIterationConfig::~MkFitIterationConfig() = default;
