#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"


#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"
#include "CommonTools/ParticleFlow/interface/PdgIdPFCandidateSelectorDefinition.h"

typedef ObjectSelector<pf2pat::PdgIdPFCandidateSelectorDefinition> PdgIdPFCandidateSelector;

template<>
void PdgIdPFCandidateSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  fillPSetSrc(desc, edm::InputTag("pfNoPileUpIso"));
  desc.add<std::vector<int> >("pdgId", std::vector<int>{22,111,130,310,2112});
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(PdgIdPFCandidateSelector);
