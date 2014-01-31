#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"

#include "FWCore/Utilities/interface/typelookup.h"

TYPELOOKUP_DATA_REG(TrajectoryFilter);

TrajectoryFilter::~TrajectoryFilter() {}
void TrajectoryFilter::setEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup) {}
