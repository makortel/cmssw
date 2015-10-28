#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"
#include "CommonTools/RecoAlgos/interface/RecoTrackViewRefSelector.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"

namespace reco {
  typedef ObjectSelector<RecoTrackViewRefSelector, edm::RefToBaseVector<reco::Track> > RecoTrackViewRefSelector;
  DEFINE_FWK_MODULE(RecoTrackViewRefSelector);
}
