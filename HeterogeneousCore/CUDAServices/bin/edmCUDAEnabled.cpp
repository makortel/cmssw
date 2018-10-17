#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

int main(void) {
  edm::MessageDrop::warningAlwaysSuppressed = true;

  auto desc = edm::ConfigurationDescriptions("Service", "CUDAService");
  CUDAService::fillDescriptions(desc);
  edm::ParameterSet ps;
  ps.addUntrackedParameter("enabled", true);
  desc.validate(ps, "CUDAService");
  edm::ActivityRegistry ar;
  auto cs = CUDAService(ps, ar);

  return cs.enabled() ? 0 : 1;
}
