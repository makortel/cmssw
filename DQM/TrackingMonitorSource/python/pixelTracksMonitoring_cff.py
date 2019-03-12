import FWCore.ParameterSet.Config as cms

import DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi
pixelTracksMonitor = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
pixelTracksMonitor.FolderName                = 'Tracking/PixelTrackParameters'
pixelTracksMonitor.TrackProducer             = 'pixelTracks'
pixelTracksMonitor.allTrackProducer          = 'pixelTracks'
pixelTracksMonitor.beamSpot                  = 'offlineBeamSpot'
pixelTracksMonitor.primaryVertex             = 'pixelVertices'
pixelTracksMonitor.pvNDOF                    = 1
pixelTracksMonitor.doAllPlots                = True
pixelTracksMonitor.doLumiAnalysis            = True
pixelTracksMonitor.doProfilesVsLS            = True
pixelTracksMonitor.doDCAPlots                = True
pixelTracksMonitor.doProfilesVsLS            = True
pixelTracksMonitor.doPlotsVsGoodPVtx         = True
pixelTracksMonitor.doEffFromHitPatternVsPU   = False
pixelTracksMonitor.doEffFromHitPatternVsBX   = False
pixelTracksMonitor.doEffFromHitPatternVsLUMI = False
pixelTracksMonitor.doPlotsVsGoodPVtx         = True
pixelTracksMonitor.doPlotsVsLUMI             = True
pixelTracksMonitor.doPlotsVsBX               = True

from CommonTools.ParticleFlow.goodOfflinePrimaryVertices_cfi import goodOfflinePrimaryVertices as _goodOfflinePrimaryVertices
goodPixelVertices = _goodOfflinePrimaryVertices.clone(
    src = "pixelVertices",
)

from DQM.TrackingMonitor.primaryVertexResolution_cfi import primaryVertexResolution as _primaryVertexResolution
pixelVertexResolution = _primaryVertexResolution.clone(
    vertexSrc = "goodPixelVertices",
    rootFolder = "OfflinePixelPV/Resolution",
)

pixelTracksMonitoringTask = cms.Task(
    goodPixelVertices
)

pixelTracksMonitoring = cms.Sequence(
    pixelTracksMonitor +
    pixelVertexResolution,
    pixelTracksMonitoringTask
)
