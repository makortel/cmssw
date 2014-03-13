#include "FWCore/Framework/interface/PluginFactoryService.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace edm {
  namespace service {
    PluginFactoryService::PluginFactoryService(const ParameterSet& procPSet) {
      std::vector<std::string> pluginNames = procPSet.getParameter<std::vector<std::string> >("@all_plugins");
      for(const auto& name: pluginNames) {
        psetMap_[name] = procPSet.getParameter<ParameterSet>(name);
      }
    }

    PluginFactoryService::~PluginFactoryService() {}

    const ParameterSet& PluginFactoryService::getPSet(const std::string& name) const {
      auto found = psetMap_.find(name);
      if(found == psetMap_.end())
        throw cms::Exception("Configuration") << "PluginFactoryService: Requested plugin '" << name << "' not found from python configuration.";
      return found->second;
    }
  }
}
