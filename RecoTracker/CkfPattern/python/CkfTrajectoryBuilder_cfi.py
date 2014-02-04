import FWCore.ParameterSet.Config as cms

import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff as trajectoryFilters

CkfTrajectoryBuilder = cms.PSet(
    ComponentType = cms.string("CkfTrajectoryBuilder"),
    propagatorAlong = cms.string('PropagatorWithMaterialParabolicMf'),
    trajectoryFilter = trajectoryFilters.CkfBaseTrajectoryFilter_block,
    maxCand = cms.int32(5),
    intermediateCleaning = cms.bool(True),
    MeasurementTrackerName = cms.string(''),
    estimator = cms.string('Chi2'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    updator = cms.string('KFUpdator'),
    alwaysUseInvalidHits = cms.bool(True),
    propagatorOpposite = cms.string('PropagatorWithMaterialParabolicMfOpposite'),
    lostHitPenalty = cms.double(30.0),
    #SharedSeedCheck = cms.bool(False)
)
