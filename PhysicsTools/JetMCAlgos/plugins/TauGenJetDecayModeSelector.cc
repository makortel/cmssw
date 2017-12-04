#include "PhysicsTools/JetMCAlgos/plugins/TauGenJetDecayModeSelector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "PhysicsTools/JetMCUtils/interface/JetMCTag.h"

TauGenJetDecayModeSelectorImp::TauGenJetDecayModeSelectorImp(const edm::ParameterSet& cfg, edm::ConsumesCollector & iC)
{
  selectedTauDecayModes_ = cfg.getParameter<vstring>("select");
}

bool TauGenJetDecayModeSelectorImp::operator()(const reco::GenJet& tauGenJet) const
{
  std::string tauGenJetDecayMode = JetMCTagUtils::genTauDecayMode(tauGenJet);
  for ( vstring::const_iterator selectedTauDecayMode = selectedTauDecayModes_.begin();
	selectedTauDecayMode != selectedTauDecayModes_.end(); ++selectedTauDecayMode ) {
    if ( tauGenJetDecayMode == (*selectedTauDecayMode) ) return true;
  }
  return false;
}

template<>
void TauGenJetDecayModeSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("tauGenJets"));

  using vstring = std::vector<std::string>;
  desc.add<vstring>("select", vstring{{"oneProng0Pi0",
                                       "oneProng1Pi0",
                                       "oneProng2Pi0",
                                       "oneProngOther",
                                       "threeProng0Pi0",
                                       "threeProng1Pi0",
                                       "threeProngOther",
                                       "rare"}});
  desc.add<bool>("filter", false);
  descriptions.add("tauGenJetsSelectorAllHadrons", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TauGenJetDecayModeSelector);
