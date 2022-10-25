#include "FWCore/Framework/interface/ModuleTypeResolverBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <string>
#include <vector>

namespace {
  void ensureValidBackendName(std::string const& backend) {
    if (not backend.empty() and backend != "serial_sync" and backend != "cuda_async") {
      edm::Exception ex(edm::errors::Configuration);
      ex << "AlpakaModuleTypeResolver was configured to use " << backend
         << " backend, but it is not supported. Currently supported backends are serial_sync, cuda_async";
      ex.addContext("Calling AlpakaModuleTypeResolverMaker constructor");
      throw ex;
    }
  }
}  // namespace

class AlpakaModuleTypeResolver : public edm::ModuleTypeResolverBase {
public:
  AlpakaModuleTypeResolver(std::string backendPrefix) : backendPrefix_(std::move(backendPrefix)) {}

  std::pair<std::string, int> resolveType(std::string basename, int index) const final {
    assert(index == kInitialIndex);
    constexpr auto kAlpaka = "@alpaka";
    auto found = basename.find(kAlpaka);
    if (found != std::string::npos) {
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
    auto backend =
        modulePSet.getUntrackedParameter<edm::ParameterSet>("alpaka").getUntrackedParameter<std::string>("backend");
    // TODO: this check is not really needed
    ensureValidBackendName(backend);
    auto prefix = fmt::format("alpaka_{}::", backend);

    LogDebug("AlpakaModuleTypeResolver")
        .format("AlpakaModuleTypeResolver: module {} backend prefix {}",
                modulePSet.getParameter<std::string>("@module_label"),
                prefix);

    return std::make_shared<AlpakaModuleTypeResolver>(prefix);
  }
};

#include "FWCore/Framework/interface/ModuleTypeResolverMakerFactory.h"
DEFINE_EDM_PLUGIN(edm::ModuleTypeResolverMakerFactory, AlpakaModuleTypeResolverMaker, "AlpakaModuleTypeResolver");
