/** \class reco::modules::GenParticleCustomSelector
 *
 *  Filter to select GenParticles
 *
 *  \author Giuseppe Cerati, UCSD
 *
 */

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/CandAlgos/interface/GenParticleCustomSelector.h"

namespace reco {
  namespace modules {
    typedef SingleObjectSelector<GenParticleCollection,::GenParticleCustomSelector> 
    GenParticleCustomSelector ;
  }
}
template <>
reco::modules::GenParticleCustomSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  fillPSetSrc(desc, edm::InputTag("genParticles"));
  desc.add<bool>("chargedOnly", true);
  desc.add<int>("status", 1);
  desc.add<std::vector<int> >("pdgId", std::vector<int>{});
  desc.add<double>("tip", 3.5);
  desc.add<double>("lip", 30.);
  desc.add<double>("ptMin", 0.9);
  desc.add<double>("minRapidity", -2.4);
  desc.add<double>("maxRapidity", 2.4);
}

namespace reco {
  namespace modules {
    DEFINE_FWK_MODULE( GenParticleCustomSelector );
  }
}
