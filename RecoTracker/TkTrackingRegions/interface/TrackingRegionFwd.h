#ifndef RecoTracker_TkTrackingRegions_TrackingRegionFwd_h
#define RecoTracker_TkTrackingRegions_TrackingRegionFwd_h

#include <memory>
#include <vector>

class TrackingRegion;
using TrackingRegionCollection = std::vector<std::unique_ptr<TrackingRegion> >;

#endif
