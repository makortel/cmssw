#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionFwd.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace RecoTracker_TkTrackingRegions {
  struct dictionary {
    std::unique_ptr<TrackingRegion> utr;
    std::vector<std::unique_ptr<TrackingRegion> > vutr;
    edm::Wrapper<std::vector<std::unique_ptr<TrackingRegion> > > wvutr;
  };
}
