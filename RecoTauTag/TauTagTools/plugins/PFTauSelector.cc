#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "CommonTools/UtilAlgos/interface/ObjectSelectorStream.h"
#include "RecoTauTag/TauTagTools/plugins/PFTauSelectorDefinition.h"

typedef ObjectSelectorStream<PFTauSelectorDefinition> PFTauSelector;
template<>
void PFTauSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("fixedConePFTauProducer"));
  desc.add<std::string>("cut", "pt > 0");

  edm::ParameterSetDescription validator;
  validator.add<edm::InputTag>("discriminator", edm::InputTag());
  validator.add<double>("selectionCut", 0.);
  edm::ParameterSet defaults;
  defaults.addParameter<edm::InputTag>("descriminator", edm::InputTag("fixedConePFTauDiscriminationByIsolation"));
  defaults.addParameter<double>("selectionCut", 0.5);
  desc.addVPSet("discriminators", validator, std::vector<edm::ParameterSet>{defaults});

  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(PFTauSelector);
