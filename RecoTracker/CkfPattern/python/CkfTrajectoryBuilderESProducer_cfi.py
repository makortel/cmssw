import FWCore.ParameterSet.Config as cms

import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff

CkfTrajectoryBuilder = cms.ESProducer("CkfTrajectoryBuilderESProducer",
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    trajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block,
    maxCand = cms.int32(5),
    ComponentName = cms.string('CkfTrajectoryBuilder'),
    intermediateCleaning = cms.bool(True),
    MeasurementTrackerName = cms.string(''),
    estimator = cms.string('Chi2'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    updator = cms.string('KFUpdator'),
    alwaysUseInvalidHits = cms.bool(True),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    lostHitPenalty = cms.double(30.0),
    #SharedSeedCheck = cms.bool(False)
)


