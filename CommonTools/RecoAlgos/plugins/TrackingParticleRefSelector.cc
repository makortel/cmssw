/** \class reco::modules::TrackingParticleSelector
 *
 *  Filter to select TrackingParticles according to pt, rapidity, tip, lip, number of hits, pdgId
 *
 *  \author Ian Tomalin, RAL
 *
 *  $Date: 2009/10/13 12:07:49 $
 *  $Revision: 1.1 $
 */

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "SimTracker/Common/interface/TrackingParticleSelector.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"

namespace reco {
  namespace modules {
    typedef SingleObjectSelector<TrackingParticleCollection,::TrackingParticleSelector,TrackingParticleRefVector> 
    TrackingParticleRefSelector ;
  }
}

// must specialize in the same namespace where ObjectSelector is defined (i.e. global)
template <>
void reco::modules::TrackingParticleRefSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  fillPSetSrc(desc, edm::InputTag("mix", "MergedTrackTruth"));
  reco::modules::ParameterAdapter<TrackingParticleSelector>::fillPSetDescription(desc);
  descriptions.addWithDefaultLabel(desc);
}

namespace reco {
  namespace modules {
    DEFINE_FWK_MODULE( TrackingParticleRefSelector );
  }
}
