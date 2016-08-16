from RecoTracker.TkTrackingRegions.globalTrackingRegionWithVertices_cfi import *
from Configuration.StandardSequences.Eras import eras
eras.trackingLowPU.toModify(globalTrackingRegionWithVertices, RegionPSet = dict(VertexCollection = "pixelVertices"))
eras.trackingPhase1PU70.toModify(globalTrackingRegionWithVertices, RegionPSet = dict(VertexCollection = "pixelVertices"))
eras.trackingPhase2PU140.toModify(globalTrackingRegionWithVertices, RegionPSet = dict(VertexCollection = "pixelVertices"))
