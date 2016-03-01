import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

from RecoLocalTracker.Configuration.RecoLocalTracker_cff import *
from SimGeneral.TrackingAnalysis.simHitTPAssociation_cfi import *
from SimTracker.TrackerHitAssociation.tpClusterProducer_cfi import *
from SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi import *
from RecoTracker.TransientTrackingRecHit.TTRHBuilders_cff import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEGeneric_cfi import *
from Geometry.TrackerNumberingBuilder.trackerTopology_cfi import *

from Validation.RecoTrack.trackingNtuple_cfi import *
from Validation.RecoTrack.TrackValidation_cff import *
import Validation.RecoTrack.TrackValidation_cff as _TrackValidation_cff

_includeHits = False
#_includeHits = True

_includeSeeds = False
#_includeSeeds = True

from PhysicsTools.RecoAlgos.trackingParticleSelector_cfi import *
trackingParticlesIntime = trackingParticleSelector.clone(
    signalOnly = False,
    intimeOnly = True,
    chargedOnly = False,
    tip = 1e5,
    lip = 1e5,
    minRapidity = -10,
    maxRapidity = 10,
    ptMin = 0,
)
simHitTPAssocProducerIntime = simHitTPAssocProducer.clone(
    trackingParticleSrc = "trackingParticlesIntime"
)
tpClusterProducerIntime = tpClusterProducer.clone(
    trackingParticleSrc = "trackingParticlesIntime"
)
quickTrackAssociatorByHitsIntime = quickTrackAssociatorByHits.clone(
    cluster2TPSrc = "tpClusterProducerIntime",
)
trackingNtuple.trackingParticles = "trackingParticlesIntime"
trackingNtuple.clusterTPMap = "tpClusterProducerIntime"
trackingNtuple.simHitTPMap = "simHitTPAssocProducerIntime"
trackingNtuple.trackAssociator = "quickTrackAssociatorByHitsIntime"
trackingNtuple.includeAllHits = _includeHits
trackingNtuple.includeSeeds = _includeSeeds

def _filterForNtuple(lst):
    ret = []
    for item in lst:
        if "PreSplitting" in item:
            continue
        if "SeedsA" in item and item.replace("SeedsA", "SeedsB") in lst:
            ret.append(item.replace("SeedsA", "Seeds"))
            continue
        if "SeedsB" in item:
            continue
        if "SeedsPair" in item and item.replace("SeedsPair", "SeedsTripl") in lst:
            ret.append(item.replace("SeedsPair", "Seeds"))
            continue
        if "SeedsTripl" in item:
            continue
        ret.append(item)
    return ret
_seedProducers = _filterForNtuple(_TrackValidation_cff._seedProducers)
_seedProducersForPhase1Pixel = _filterForNtuple(_TrackValidation_cff._seedProducersForPhase1Pixel)

(_seedSelectors, trackingNtupleSeedSelectors) = _TrackValidation_cff._addSeedToTrackProducers(_seedProducers, globals())
(_phase1PixelSeedSelectors_, _phase1PixelTrackingNtupleSeedSelectors) = _TrackValidation_cff._addSeedToTrackProducers(_seedProducersForPhase1Pixel, globals())
eras.phase1Pixel.toReplaceWith(trackingNtupleSeedSelectors, _phase1PixelTrackingNtupleSeedSelectors)

trackingNtuple.seedTracks = _seedSelectors
eras.phase1Pixel.toModify(trackingNtuple, seedTracks = _seedProducersForPhase1Pixel)

trackingNtupleSequence = cms.Sequence()
# reproduce hits because they're not stored in RECO
if _includeHits:
    trackingNtupleSequence += (
        siPixelRecHits +
        siStripMatchedRecHits
    )
if _includeSeeds:
    trackingNtupleSequence += trackingNtupleSeedSelectors

trackingNtupleSequence += (
    # sim information
    trackingParticlesIntime +
    simHitTPAssocProducerIntime +
    tpClusterProducerIntime +
    quickTrackAssociatorByHitsIntime +
    # ntuplizer
    trackingNtuple
)
