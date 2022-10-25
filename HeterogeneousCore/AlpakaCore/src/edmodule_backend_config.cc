#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/edmodule_backend_config.h"

namespace {
  const std::string kPSetName("alpaka");
  const char* const kComment = "PSet contains the possibility to override the Alpaka backend per module instance";
}  // namespace

namespace cms::alpakatools {
  void edmodule_backend_config(edm::ConfigurationDescriptions& iDesc) {
    // NOTE: by not giving a default, we are intentionally not having 'alpaka' added
    // to any cfi files. TODO: I don't know if HLT would want them or not. The default would be an empty PSet.
    edm::ParameterSetDescription descAlpaka;
    descAlpaka.addUntracked<std::string>("backend", "")
        ->setComment(
            "Alpaka backend for this module. Can be empty string (for the global default), 'serial_sync', or "
            "'cuda_async'");
    //descAlpaka.addOptionalUntracked<std::string>("backend")->setComment("Alpaka backend for this module. Can be empty string (for the global default), 'serial_sync', or 'cuda_async'");

    if (iDesc.defaultDescription()) {
      if (iDesc.defaultDescription()->isLabelUnused(kPSetName)) {
        //iDesc.defaultDescription()->addOptionalUntracked<edm::ParameterSetDescription>(kPSetName, descAlpaka)->setComment(kComment);
        iDesc.defaultDescription()
            ->addUntracked<edm::ParameterSetDescription>(kPSetName, descAlpaka)
            ->setComment(kComment);
      }
    }
    for (auto& v : iDesc) {
      if (v.second.isLabelUnused(kPSetName)) {
        v.second.addUntracked<edm::ParameterSetDescription>(kPSetName, descAlpaka)->setComment(kComment);
        //v.second.addOptionalUntracked<edm::ParameterSetDescription>(kPSetName, descAlpaka)->setComment(kComment);
        //v.second.addVPSetUntracked("foo", descAlpaka, std::vector<edm::ParameterSet>());
      }
    }
  }
}  // namespace cms::alpakatools
