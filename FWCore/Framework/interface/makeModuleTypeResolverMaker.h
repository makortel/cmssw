#ifndef FWCore_Framework_makeModuleTypeResolverMaker_h
#define FWCore_Framework_makeModuleTypeResolverMaker_h

#include <memory>

#include "FWCore/Framework/interface/ModuleTypeResolverBase.h"

namespace edm {
  class ParameterSet;

  std::unique_ptr<edm::ModuleTypeResolverMaker> makeModuleTypeResolverMaker(edm::ParameterSet const& pset);
}  // namespace edm

#endif
