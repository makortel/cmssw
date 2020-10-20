#include "CondFormats/DataRecord/interface/SiPixelDynamicInefficiencyMixingRcd.h"
#include "CondFormats/RunInfo/interface/MixingModuleConfig.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelDynamicInefficiency.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Utilities/interface/ESProductTag.h"
#include "FWCore/Utilities/interface/do_nothing_deleter.h"

class SiPixelDynamicInefficiencyChoiceByBunchSpace : public edm::ESProducer {
public:
  SiPixelDynamicInefficiencyChoiceByBunchSpace(edm::ParameterSet const& iConfig);
  ~SiPixelDynamicInefficiencyChoiceByBunchSpace() = default;

  std::shared_ptr<SiPixelDynamicInefficiency const> produce(SiPixelDynamicInefficiencyMixingRcd const& iRecord);

private:
  edm::ESGetToken<SiPixelDynamicInefficiency, SiPixelDynamicInefficiencyRcd> token_;
};

SiPixelDynamicInefficiencyChoiceByBunchSpace::SiPixelDynamicInefficiencyChoiceByBunchSpace(
    edm::ParameterSet const& iConfig) {
  setWhatProduced(this).setMayConsume(
      token_,
      [](auto const& get, edm::ESTransientHandle<MixingModuleConfig> config) {
        if (config->bunchSpace() == 50) {
          return get("", "50ns");
        } else {
          return get("", "");
        }
      },
      edm::ESProductTag<MixingModuleConfig, MixingRcd>("", ""));
}

std::shared_ptr<SiPixelDynamicInefficiency const> SiPixelDynamicInefficiencyChoiceByBunchSpace::produce(
    SiPixelDynamicInefficiencyMixingRcd const& iRecord) {
  return std::shared_ptr<SiPixelDynamicInefficiency const>(&iRecord.get(token_), edm::do_nothing_deleter());
}

DEFINE_FWK_EVENTSETUP_MODULE(SiPixelDynamicInefficiencyChoiceByBunchSpace);
