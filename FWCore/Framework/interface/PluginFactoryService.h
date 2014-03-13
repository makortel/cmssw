#ifndef FWCore_Framework_PluginFactoryService
#define FWCore_Framework_PluginFactoryService

#include<string>
#include<map>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
  namespace service {
    class PluginFactoryService {
    public:
      explicit PluginFactoryService(const ParameterSet& procPSet);
      ~PluginFactoryService();

      template <typename FactoryType, typename... Args>
      auto create(const std::string& name, Args... args) -> typename FactoryType::ReturnType const {
        const ParameterSet& pset = getPSet(name);
        return FactoryType::get()->create(pset.getParameter<std::string>("@module_type"), pset, std::forward<Args>(args)...);
      }

    private:
      const ParameterSet& getPSet(const std::string& name) const;
 
      std::map<std::string, ParameterSet> psetMap_;
    };
  }
}

#endif
