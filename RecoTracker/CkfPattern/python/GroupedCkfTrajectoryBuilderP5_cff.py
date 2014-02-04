import FWCore.ParameterSet.Config as cms

# initialize magnetic field #########################

# initialize geometry #####################

# KFUpdatoerESProducer
from TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi import *
# Chi2MeasurementEstimatorESProducer
from TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi import *
# PropagatorWithMaterialESProducer
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
# PropagatorWithMaterialESProducer
from TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi import *
# stripCPE
from RecoLocalTracker.SiStripRecHitConverter.StripCPE_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
# pixelCPE
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
#TransientTrackingBuilder
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi

## MeasurementTracker
##CTF_P5_MeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone()
#replace CTF_P5_MeasurementTracker.pixelClusterProducer = ""

# trajectory filtering
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
ckfBaseTrajectoryFilterP5 = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minPt = 0.5,
    maxLostHits = 4,
    maxConsecLostHits = 3
)
#replace ckfBaseTrajectoryFilterP5.filterPset.minimumNumberOfHits =  4
#
GroupedCkfTrajectoryBuilderP5 = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone()
##CTF_P5_MeasurementTracker.ComponentName = 'CTF_P5' # useless duplication of MeasurementTracker
##GroupedCkfTrajectoryBuilderP5.MeasurementTrackerName = 'CTF_P5' # useless duplication of MeasurementTracker
GroupedCkfTrajectoryBuilderP5.trajectoryFilter = ckfBaseTrajectoryFilterP5

