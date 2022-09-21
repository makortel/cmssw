#ifndef FWCore_Framework_itnerface_ModuleTypeResolverFactory_h
#define FWCore_Framework_itnerface_ModuleTypeResolverFactory_h

#include "FWCore/PluginManager/interface/PluginFactory.h"

#include <string>
#include <vector>

namespace edm {
  class ModuleTypeResolverMaker;
  class ParameterSet;

  using ModuleTypeResolverMakerFactory = edmplugin::PluginFactory<ModuleTypeResolverMaker*(edm::ParameterSet const&)>;
}  // namespace edm

#endif
