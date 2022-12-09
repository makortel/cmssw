#include "FWCore/Framework/interface/ModuleTypeResolverBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <string>
#include <vector>

class AlpakaModuleTypeResolver : public edm::ModuleTypeResolverBase {
public:
  AlpakaModuleTypeResolver(std::string backendPrefix) : backendPrefix_(std::move(backendPrefix)) {}

  std::pair<std::string, int> resolveType(std::string basename, int index) const final {
    assert(index == kInitialIndex);
    constexpr auto kAlpaka = "@alpaka";
    auto found = basename.find(kAlpaka);
    if (found != std::string::npos) {
      if (backendPrefix_.empty()) {
        throw edm::Exception(edm::errors::LogicError) << "AlpakaModuleTypeResolver encountered a module with type name " << basename << " but the backend prefix was empty. This should not happen. Please contact framework developers.";
      }
      basename.replace(found, std::strlen(kAlpaka), "");
      basename = backendPrefix_ + basename;
    }
    return {basename, kLastIndex};
  }

private:
  std::string const backendPrefix_;
};

class AlpakaModuleTypeResolverMaker : public edm::ModuleTypeResolverMaker {
public:
  AlpakaModuleTypeResolverMaker(edm::ParameterSet const& iConfig) {}

  std::shared_ptr<edm::ModuleTypeResolverBase const> makeResolver(edm::ParameterSet const& modulePSet) const final {
    std::string prefix = "";
    if (modulePSet.existsAs<edm::ParameterSet>("alpaka", false)) {
      auto const& backend =
        modulePSet.getUntrackedParameter<edm::ParameterSet>("alpaka").getUntrackedParameter<std::string>("backend");
      prefix = fmt::format("alpaka_{}::", backend);

      LogDebug("AlpakaModuleTypeResolver")
        .format("AlpakaModuleTypeResolver: module {} backend prefix {}",
                modulePSet.getParameter<std::string>("@module_label"),
                prefix);

    }
    return std::make_shared<AlpakaModuleTypeResolver>(prefix);
  }
};

#include "FWCore/Framework/interface/ModuleTypeResolverMakerFactory.h"
DEFINE_EDM_PLUGIN(edm::ModuleTypeResolverMakerFactory, AlpakaModuleTypeResolverMaker, "AlpakaModuleTypeResolver");
