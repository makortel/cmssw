/** \class reco::modules::TrackingParticleSelector
 *
 *  Filter to select TrackingParticles according to pt, rapidity, tip, lip, number of hits, pdgId
 *
 *  \author Giuseppe Cerati, INFN
 *
 *  $Date: 2009/03/04 13:11:31 $
 *  $Revision: 1.1 $
 */

#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "SimTracker/Common/interface/TrackingParticleSelector.h"

namespace reco {
  namespace modules {
    typedef SingleObjectSelector<TrackingParticleCollection,::TrackingParticleSelector> 
    TrackingParticleSelector ;
  }
}

template <>
void reco::modules::TrackingParticleSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  fillPSetSrc(desc, edm::InputTag("mix", "MergedTrackTruth"));
  reco::modules::ParameterAdapter<TrackingParticleSelector>::fillPSetDescription(desc);
  descriptions.addWithDefaultLabel(desc);
}

namespace reco {
  namespace modules {
    DEFINE_FWK_MODULE( TrackingParticleSelector );
  }
}

