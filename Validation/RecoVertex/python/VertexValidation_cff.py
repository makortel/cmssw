import FWCore.ParameterSet.Config as cms

from SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cfi import *
from SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi import *
from Validation.RecoVertex.v0validator_cfi import *
from Validation.RecoVertex.PrimaryVertexAnalyzer4PUSlimmed_cfi import *

# Rely on tracksValidationTruth sequence being already run
vertexValidation = cms.Sequence(v0Validator
                                * vertexAnalysisSequence)
