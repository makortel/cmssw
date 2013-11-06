#ifndef _ClusterShapeTrajectoryFilter_h_
#define _ClusterShapeTrajectoryFilter_h_

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"

namespace edm { class ParameterSet; class EventSetup; }

class SiPixelRecHit;
class SiStripRecHit2D;
class GlobalTrackingGeometry;
class MagneticField;
class SiPixelLorentzAngle;
class SiStripLorentzAngle;
class ClusterShapeHitFilter;

class ClusterShapeTrajectoryFilter : public TrajectoryFilter {
 public:
  //  ClusterShapeTrajectoryFilter(const edm::EventSetup& es);

  ClusterShapeTrajectoryFilter(const edm::ParameterSet& iConfig, edm::ConsumesCollector& iC): theFilter(nullptr) {}
  ClusterShapeTrajectoryFilter
    (const ClusterShapeHitFilter * f):theFilter(f){}

  virtual ~ClusterShapeTrajectoryFilter();

  ClusterShapeTrajectoryFilter *clone(const edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  virtual bool qualityFilter(const TempTrajectory&) const;
  virtual bool qualityFilter(const Trajectory&) const;
 
  virtual bool toBeContinued(TempTrajectory&) const;
  virtual bool toBeContinued(Trajectory&) const;

  virtual std::string name() const { return "ClusterShapeTrajectoryFilter"; }

 private:

  const ClusterShapeHitFilter * theFilter;
};

#endif
