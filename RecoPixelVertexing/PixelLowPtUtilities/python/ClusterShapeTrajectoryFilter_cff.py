import FWCore.ParameterSet.Config as cms

ClusterShapeTrajectoryFilter_block = cms.PSet(
    ComponentType = cms.string('ClusterShapeTrajectoryFilter'),
    cacheSrc = cms.InputTag('siPixelClusterShapeCache'),
)
