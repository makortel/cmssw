import FWCore.ParameterSet.Config as cms

quickTrackAssociatorByHits = cms.EDProducer("QuickTrackAssociatorByHitsProducer",
	AbsoluteNumberOfHits = cms.bool(False),
	Quality_RecoToSim = cms.double(0.75),
	Purity_RecoToSim = cms.double(0.75),
	RecoToSimDenominator = cms.string('reco'), # either "reco" or "recoOrSim"
	SimToRecoDenominator = cms.string('reco'), # either "sim", "reco", or "recoOrSim"
	Quality_SimToReco = cms.double(0.5),
	Purity_SimToReco = cms.double(0.75),
	ThreeHitTracksAreSpecial = cms.bool(True),
        PixelHitWeight = cms.double(1.0),
	associatePixel = cms.bool(True),
	associateStrip = cms.bool(True),
        pixelSimLinkSrc = cms.InputTag("simSiPixelDigis"),
        stripSimLinkSrc = cms.InputTag("simSiStripDigis"),
        useClusterTPAssociation = cms.bool(True),
        cluster2TPSrc = cms.InputTag("tpClusterProducer")
)

from Configuration.StandardSequences.Eras import eras
if eras.fastSim.isChosen():
    quickTrackAssociatorByHits.associateStrip = False
    quickTrackAssociatorByHits.associatePixel = False
    quickTrackAssociatorByHits.useClusterTPAssociation = False
