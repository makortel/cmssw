import FWCore.ParameterSet.Config as cms
from RecoPixelVertexing.PixelTriplets.caHitQuadrupletEDProducer_cfi import caHitQuadrupletEDProducer as _caHitQuadrupletEDProducer

def customizeLegacy(process, module):
    pset = getattr(module, "OrderedHitsFactoryPSet")
    if not hasattr(pset, "ComponentName"):
        return
    if not (pset.ComponentName == "CombinedHitQuadrupletGenerator"):
        return
    # Adjust seeding layers
    seedingLayersName = module.OrderedHitsFactoryPSet.SeedingLayers.getModuleLabel()



    # Configure seed generator / pixel track producer
    quadruplets = module.OrderedHitsFactoryPSet.clone()
    from RecoPixelVertexing.PixelTriplets.CAHitQuadrupletGenerator_cfi import CAHitQuadrupletGenerator as _CAHitQuadrupletGenerator

    module.OrderedHitsFactoryPSet  = _CAHitQuadrupletGenerator.clone(
        ComponentName = cms.string("CAHitQuadrupletGenerator"),
        extraHitRPhitolerance = quadruplets.GeneratorPSet.extraHitRPhitolerance,
        maxChi2 = dict(
            pt1    = 0.8, pt2    = 2,
            value1 = 200, value2 = 100,
            enabled = True,
        ),
        useBendingCorrection = True,
        fitFastCircle = True,
        fitFastCircleChi2Cut = True,
        SeedingLayers = cms.InputTag(seedingLayersName),
        CAThetaCut = cms.double(0.00125),
        CAPhiCut = cms.double(10),
    )

    if hasattr(quadruplets.GeneratorPSet, "SeedComparitorPSet"):
        pset.SeedComparitorPSet = quadruplets.GeneratorPSet.SeedComparitorPSet

def customizeNew(process, module):
    # Bit of a hack to replace a module with another, but works
    modifier = cms.Modifier()
    modifier._setChosen()

    tripletProducer = getattr(process, module.triplets.getModuleLabel())

    modifier.toReplaceWith(module, _caHitQuadrupletEDProducer.clone(
        doublets = tripletProducer.doublets.value(),
        extraHitRPhitolerance = module.extraHitRPhitolerance.value(),
        maxChi2 = dict(
            pt1    = 0.8, pt2    = 2,
            value1 = 200, value2 = 100,
            enabled = True,
        ),
        useBendingCorrection = True,
        fitFastCircle = True,
        fitFastCircleChi2Cut = True,
        CAThetaCut = 0.00125,
        CAPhiCut = 10,
    ))

    for seqName, seq in process.sequences_().iteritems():
        seq.remove(tripletProducer)

def customiseForQuadrupletsByCellularAutomaton(process):
    for module in process._Process__producers.values():
        if hasattr(module, "OrderedHitsFactoryPSet"):
            customizeLegacy(process, module)
        elif module._TypedParameterizable__type == "PixelQuadrupletEDProducer":
            customizeNew(process, module)
    return process
