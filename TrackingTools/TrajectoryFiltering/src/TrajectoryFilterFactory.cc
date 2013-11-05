#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilterFactory.h"

EDM_REGISTER_PLUGINFACTORY(TrajectoryFilterFactory, "TrajectoryFilterFactory");

TrajectoryFilter *createTrajectoryFilter(const edm::ParameterSet& pset, const std::string& psetName, edm::ConsumesCollector& iC) {
  if(pset.exists(psetName)) {
    const edm::ParameterSet& p = pset.getParameter<edm::ParameterSet>(psetName);
    return TrajectoryFilterFactory::get()->create(p.getParameter<std::string>("ComponentType"), p);
  }
  return nullptr;
}
