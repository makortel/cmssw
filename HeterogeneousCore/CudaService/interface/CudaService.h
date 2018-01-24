#ifndef HeterogeneousCore_CudaService_CudaService_h
#define HeterogeneousCore_CudaService_CudaService_h

#include <utility>
#include <vector>

namespace edm {
  class ParameterSet;
  class ActivityRegistry;
}

class CudaService {
public:
  CudaService(edm::ParameterSet const& iConfig, edm::ActivityRegistry& iRegistry);

  bool enabled() const { return enabled_; }

  int numberOfDevices() const { return numberOfDevices_; }

  // major, minor
  std::pair<int, int> computeCapability(int device) { return computeCapabilities_.at(device); }

private:
  int numberOfDevices_ = 0;
  std::vector<std::pair<int, int> > computeCapabilities_;
  bool enabled_ = false;
};

#endif
