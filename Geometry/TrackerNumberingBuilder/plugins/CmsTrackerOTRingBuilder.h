#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerOTRingBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerOTRingBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>
/**
 * Class which contructs PixelForward Panels. 
 */
class CmsTrackerOTRingBuilder : public CmsTrackerLevelBuilder {
private:
  void sortNS(DDFilteredView&, GeometricDet*) override;
  void buildComponent(DDFilteredView&, GeometricDet*, std::string) override;
};

#endif
