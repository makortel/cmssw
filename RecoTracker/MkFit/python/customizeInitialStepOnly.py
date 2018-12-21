import FWCore.ParameterSet.Config as cms

def removePath(process, pname):
    if hasattr(process, pname):
        process.schedule.remove(getattr(process, pname))
        delattr(process, pname)

def customizeInitialStepOnly(process):
    # Customize reconstruction
    process.trackerClusterCheck.PixelClusterCollectionLabel = 'siPixelClustersPreSplitting'
    process.initialStepSeedLayers.FPix.HitProducer = 'siPixelRecHitsPreSplitting'
    process.initialStepSeedLayers.BPix.HitProducer = 'siPixelRecHitsPreSplitting'
    process.initialStepHitQuadruplets.SeedComparitorPSet.clusterShapeCacheSrc = "siPixelClusterShapeCachePreSplitting"
    process.initialStepSeeds.SeedComparitorPSet.ClusterShapeCacheSrc = "siPixelClusterShapeCachePreSplitting"
    if hasattr(process.initialStepTrackCandidates, "measurementTrackerEvent"):
        process.initialStepTrackCandidates.measurementTrackerEvent = 'MeasurementTrackerEventPreSplitting'
        process.initialStepTrackCandidates.pixelRecHits = "siPixelRecHitsPreSplitting"
    else:
        process.initialStepTrackCandidates.MeasurementTrackerEvent = 'MeasurementTrackerEventPreSplitting'
    process.initialStepTracks.MeasurementTrackerEvent = 'MeasurementTrackerEventPreSplitting'
    process.iterTrackingTask = cms.Task(process.trackerClusterCheck,
                                        process.InitialStepTask)

    # Customize MTV
    for selector in [
        process.cutsRecoTracksInitialStep,
        process.cutsRecoTracksPt09InitialStep,
        process.cutsRecoTracksFromPVInitialStep,
        process.cutsRecoTracksFromPVPt09InitialStep,
        ]:
        selector.algorithm = []
        selector.src = "initialStepTracks"
        selector.vertexTag = "firstStepPrimaryVertices"

    process.trackingParticleRecoTrackAsssociation.label_tr = "initialStepTracks"

    def setInput(mtvs, labels):
        for mtv in mtvs:
            mod = getattr(process, mtv)
            mod.label = labels
            mod.label_vertex = "firstStepPrimaryVertices"
            mod.trackCollectionForDrCalculation = "initialStepTracks"
            mod.dodEdxPlots = False

    setInput(["trackValidatorTrackingOnly", "trackValidatorAllTPEfficStandalone", "trackValidatorTPPtLess09Standalone", "trackValidatorBHadronTrackingOnly"],
             ["cutsRecoTracksInitialStep", "cutsRecoTracksPt09InitialStep"])
    setInput(["trackValidatorFromPVStandalone", "trackValidatorFromPVAllTPStandalone"], ["cutsRecoTracksFromPVInitialStep", "cutsRecoTracksFromPVPt09InitialStep"])
    setInput(["trackValidatorSeedingTrackingOnly"], ["seedTracksinitialStepSeeds"])
    setInput(["trackValidatorBuilding"], ["initialStepTracks"])
    process.trackValidatorBuilding.mvaLabels = cms.untracked.PSet(initialStepTracks = cms.untracked.vstring('initialStep'))

    process.tracksPreValidationTrackingOnly = cms.Sequence(
        process.cutsRecoTracksInitialStep+
        process.cutsRecoTracksPt09InitialStep+
        process.cutsRecoTracksFromPVInitialStep+
        process.cutsRecoTracksFromPVPt09InitialStep+
        process.tracksValidationTruth+
        cms.ignore(process.trackingParticlesSignal)+
        process.trackingParticlesBHadron
    )
    process.trackValidatorsTrackingOnly.remove(process.trackValidatorConversionTrackingOnly)

    # Remove vertex validation
    process.globalPrevalidationTrackingOnly.remove(process.vertexValidationTrackingOnly)

    # Remove DQM
    removePath(process, "dqmoffline_step")
    removePath(process, "dqmofflineOnPAT_step")

    # Remove RECO output if it exists
    removePath(process, "RECOSIMoutput_step")

    return process
