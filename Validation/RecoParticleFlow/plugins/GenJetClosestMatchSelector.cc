#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"
#include "Validation/RecoParticleFlow/plugins/GenJetClosestMatchSelectorDefinition.h"

typedef ObjectSelector<GenJetClosestMatchSelectorDefinition> GenJetClosestMatchSelector;
template<>
void GenJetClosestMatchSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("ak4GenJets"));
  desc.add<edm::InputTag>("MatchTo", edm::InputTag("tauGenJetsSelectorAllHadrons"));
  desc.add<bool>("filter", false);
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(GenJetClosestMatchSelector);

