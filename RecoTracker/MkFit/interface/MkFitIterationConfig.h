#ifndef RecoTracker_MkFit_MkFitIterationConfig_h
#define RecoTracker_MkFit_MkFitIterationConfig_h

#include <memory>

namespace mkfit {
  class IterationsInfo;
  class IterationConfig;
}  // namespace mkfit

/**
 * Holds per-iteration parameters of mkFit for one iteration. An
 * ESProduct avoids the need to have per-stream copies of the
 * parameters.
 */
class MkFitIterationConfig {
public:
  MkFitIterationConfig(std::unique_ptr<mkfit::IterationsInfo> info, const mkfit::IterationConfig* config);
  ~MkFitIterationConfig();

  const mkfit::IterationConfig& get() const { return *config_; }

private:
  std::unique_ptr<mkfit::IterationsInfo> iterationsInfo_;  // copy of all parameters
  const mkfit::IterationConfig* config_;  // parameters of this iteration, points to an object inside iterationsInfo_
};

#endif
