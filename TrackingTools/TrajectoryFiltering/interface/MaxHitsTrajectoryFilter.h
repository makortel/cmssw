#ifndef MaxHitsTrajectoryFilter_H
#define MaxHitsTrajectoryFilter_H

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"

class MaxHitsTrajectoryFilter : public TrajectoryFilter {
public:

  explicit MaxHitsTrajectoryFilter( int maxHits=-1): theMaxHits( maxHits) {}
    
  explicit MaxHitsTrajectoryFilter(const edm::ParameterSet & pset, edm::ConsumesCollector& iC):
    theMaxHits( pset.getParameter<int>("maxNumberOfHits")) {}

  MaxHitsTrajectoryFilter *clone(const edm::Event& iEvent, const edm::EventSetup& iSetup) const override {
    return new MaxHitsTrajectoryFilter(*this);
  }
  
  virtual bool qualityFilter( const Trajectory& traj) const { return TrajectoryFilter::qualityFilterIfNotContributing; }
  virtual bool qualityFilter( const TempTrajectory& traj) const { return TrajectoryFilter::qualityFilterIfNotContributing; }

  virtual bool toBeContinued( TempTrajectory& traj) const {return TBC<TempTrajectory>(traj);}
  virtual bool toBeContinued( Trajectory& traj) const { return TBC<Trajectory>(traj);}

  virtual std::string name() const {return "MaxHitsTrajectoryFilter";}

 protected:

  template<class T> bool TBC(const T & traj) const{
    if ( (traj.foundHits() < theMaxHits) || theMaxHits<0) return true;
    else return false;
  }

  float theMaxHits;

};

#endif
