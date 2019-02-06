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
        # MkFit case
        process.initialStepTrackCandidates.measurementTrackerEvent = 'MeasurementTrackerEventPreSplitting'
        process.initialStepTrackCandidatesMkFitInput.pixelRecHits = "siPixelRecHitsPreSplitting"
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

    process.trackingParticleRecoTrackAsssociationPreSplitting = process.trackingParticleRecoTrackAsssociation.clone(
        label_tr = "initialStepTracks",
        associator = "quickTrackAssociatorByHitsPreSplitting",
    )
    process.VertexAssociatorByPositionAndTracksPreSplitting = process.VertexAssociatorByPositionAndTracks.clone(
        trackAssociation = "trackingParticleRecoTrackAsssociationPreSplitting"
    )


    def setInput(mtvs, labels):
        for mtv in mtvs:
            mod = getattr(process, mtv)
            mod.label = labels
            mod.label_vertex = "firstStepPrimaryVertices"
            if mod.UseAssociators.value():
                mod.associators = ["quickTrackAssociatorByHitsPreSplitting"]
            else:
                mod.associators = ["trackingParticleRecoTrackAsssociationPreSplitting"]
            mod.vertexAssociator = "VertexAssociatorByPositionAndTracksPreSplitting"
            mod.trackCollectionForDrCalculation = "initialStepTracks"
            mod.dodEdxPlots = False

    setInput(["trackValidatorTrackingOnly", "trackValidatorAllTPEfficStandalone", "trackValidatorTPPtLess09Standalone", "trackValidatorBHadronTrackingOnly"],
             ["cutsRecoTracksInitialStep", "cutsRecoTracksPt09InitialStep"])
    setInput(["trackValidatorFromPVStandalone", "trackValidatorFromPVAllTPStandalone"], ["cutsRecoTracksFromPVInitialStep", "cutsRecoTracksFromPVPt09InitialStep"])
    setInput(["trackValidatorSeedingTrackingOnly"], ["seedTracksinitialStepSeeds"])
    setInput(["trackValidatorBuilding"], ["initialStepTracks"])
    process.trackValidatorBuilding.mvaLabels = cms.untracked.PSet(initialStepTracks = cms.untracked.vstring('initialStep'))

    process.tracksPreValidationTrackingOnly = cms.Task(
        process.cutsRecoTracksInitialStep,
        process.cutsRecoTracksPt09InitialStep,
        process.cutsRecoTracksFromPVInitialStep,
        process.cutsRecoTracksFromPVPt09InitialStep,
        process.tracksValidationTruth,
        process.trackingParticlesSignal,
        process.trackingParticlesBHadron,
        process.trackingParticleRecoTrackAsssociationPreSplitting,
        process.VertexAssociatorByPositionAndTracksPreSplitting,
    )
    process.trackValidatorsTrackingOnly.remove(process.trackValidatorConversionTrackingOnly)
    process.trackValidatorsTrackingOnly.remove(process.trackValidatorSeedingPreSplittingTrackingOnly)
    process.trackValidatorsTrackingOnly.remove(process.trackValidatorBuildingPreSplitting)

    # Remove vertex validation
    process.globalPrevalidationTrackingOnly.remove(process.vertexValidationTrackingOnly)

    # Remove DQM
    removePath(process, "dqmoffline_step")
    removePath(process, "dqmofflineOnPAT_step")

    # Remove RECO output if it exists
    removePath(process, "RECOSIMoutput_step")

    return process


def customizeInitialStepOnlyNoMTV(process):
    process = customizeInitialStepOnly(process)

    # Remove validation
    removePath(process, "prevalidation_step")
    removePath(process, "validation_step")
    removePath(process, "DQMoutput_step")

    # Add a dummy output module to trigger the (minimal) prefetching
    process.out = cms.OutputModule("AsciiOutputModule",
        outputCommands = cms.untracked.vstring(
            "keep *_initialStepTracks_*_*",
        ),
        verbosity = cms.untracked.uint32(0)
    )
    process.outPath = cms.EndPath(process.out)
    process.schedule = cms.Schedule(process.raw2digi_step, process.reconstruction_step, process.outPath)

    return process
