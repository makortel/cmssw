import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelTrackFitting.pixelTracks_cff import *
from RecoMuon.L3MuonIsolationProducer.isolationRegionAroundL3Muon_cfi import isolationRegionAroundL3Muon

pixelTracksForL3IsolationHitDoublets = pixelTracksHitDoublets.clone(
    trackingRegions = "isolationRegionAroundL3Muon"
)
pixelTracksForL3IsolationHitTriplets = pixelTracksHitTriplets.clone(
    hitDoublets = "pixelTracksForL3IsolationHitDoublets"
)

pixelTracksForL3Isolation = pixelTracks.clone(
    SeedingHitSets = "pixelTracksForL3IsolationHitTriplets"
)

# 2016-11-28 MK I assume this is not needed for phase2
pixelTracksForL3IsolationSequence = cms.Sequence(
    isolationRegionAroundL3Muon + 
    pixelTracksForL3IsolationHitDoublets +
    pixelTracksForL3IsolationHitTriplets +
    pixelTracksForL3Isolation
)
