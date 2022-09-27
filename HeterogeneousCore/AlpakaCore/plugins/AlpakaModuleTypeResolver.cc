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
  constexpr auto kCPU = "cpu";
  constexpr auto kNvidiaGPU = "gpu-nvidia";

  bool contains(std::vector<std::string> const& vec, std::string const& acc) {
    return std::find(vec.begin(), vec.end(), acc) != vec.end();
  }

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
  AlpakaModuleTypeResolverMaker(edm::ParameterSet const& iConfig,
                                std::vector<std::string> const& selectedAccelerators) {
    auto const& backend = iConfig.getUntrackedParameter<std::string>("backend");
    ensureValidBackendName(backend);

    if (contains(selectedAccelerators, kNvidiaGPU)) {
      availableBackends_.push_back("cuda_async");
    }
    if (contains(selectedAccelerators, kCPU)) {
      availableBackends_.push_back("serial_sync");
    }
    if (not backend.empty()) {
      auto found = std::find(availableBackends_.begin(), availableBackends_.end(), backend);
      if (found == availableBackends_.end()) {
        edm::Exception ex(edm::errors::UnavailableAccelerator);
        ex << "AlpakaModuleTypeResolver was configured to use " << backend
           << " backend, but the job does not have the required accelerator";
        ex.addContext("Calling AlpakaModuleTypeResolverMaker constructor");
        throw ex;
      }
      if (found != availableBackends_.begin()) {
        std::rotate(availableBackends_.begin(), found, found - 1);
      }
    }

    if (not availableBackends_.empty()) {
      LogDebug("AlpakaModuleTypeResolver")
          .format("AlpakaModuleTypeResolver: global default backend prefix {}", availableBackends_.front());
    } else {
      LogDebug("AlpakaModuleTypeResolver").format("AlpakaModuleTypeResolver: no global default backend prefix");
    }
  }

  std::shared_ptr<edm::ModuleTypeResolverBase const> makeResolver(edm::ParameterSet const& modulePSet) const final {
#ifdef NOTYET
    auto backend =
        modulePSet.getUntrackedParameter<edm::ParameterSet>("alpaka").getUntrackedParameter<std::string>("backend");
    if (backend.empty()) {
      if (availableBackends_.empty()) {
        edm::Exception ex(edm::errors::UnavailableAccelerator);
        ex << "AlpakaModuleTypeResolver had no backends available because of the combination of job configuration and "
              "accelerator availability on the machine";
        ex.addContext("Calling AlpakaModuleTypeResolverMaker::makeResolver()");
        throw ex;
      }
      backend = availableBackends_.front();
    } else {
      ensureValidBackendName(backend);
      if (not contains(availableBackends_, backend)) {
        edm::Exception ex(edm::errors::UnavailableAccelerator);
        ex << "AlpakaModuleTypeResolver was configured to use backend " << backend << " for module "
           << modulePSet.getParameter<std::string>("@module_label")
           << ", but that backend is not available for job because of the combination of job configuration and "
              "accelerator availability on the machine";
        ex.addContext("Calling AlpakaModuleTypeResolverMaker::makeResolver()");
        throw ex;
      }
    }
#else
    if (availableBackends_.empty()) {
      if (availableBackends_.empty()) {
        edm::Exception ex(edm::errors::UnavailableAccelerator);
        ex << "AlpakaModuleTypeResolver had no backends available because of the combination of job configuration and "
              "accelerator availability on the machine";
        ex.addContext("Calling AlpakaModuleTypeResolverMaker::makeResolver()");
        throw ex;
      }
    }
    auto const& backend = availableBackends_.front();
#endif
    auto prefix = fmt::format("alpaka_{}::", backend);

    LogDebug("AlpakaModuleTypeResolver")
        .format("AlpakaModuleTypeResolver: module {} backend prefix {}",
                modulePSet.getParameter<std::string>("@module_label"),
                prefix);

    return std::make_shared<AlpakaModuleTypeResolver>(prefix);
  }

private:
  // vector of backends available in the job, first element is the
  // backend to be used by default
  std::vector<std::string> availableBackends_;
};

#include "FWCore/Framework/interface/ModuleTypeResolverMakerFactory.h"
DEFINE_EDM_PLUGIN(edm::ModuleTypeResolverMakerFactory, AlpakaModuleTypeResolverMaker, "AlpakaModuleTypeResolver");
