from HLTrigger.Configuration.common import replace_with

def customiseExcludeBPix1(process):
    # InitialStep(PreSplitting) to triplets
    for postfix in ["PreSplitting", ""]:
        getattr(process, "initialStepSeedLayers"+postfix).layerList = [
            'BPix2+BPix3+BPix4',
            'BPix2+BPix3+FPix1_pos',
            'BPix2+BPix3+FPix1_neg',
            'BPix2+FPix1_pos+FPix2_pos',
            'BPix2+FPix1_neg+FPix2_neg',
            'FPix1_pos+FPix2_pos+FPix3_pos',
            'FPix1_neg+FPix2_neg+FPix3_neg',
        ]

        getattr(process, "initialStepHitDoublets"+postfix).layerPairs = [0,1] # layer pairs (0,1), (1,2)

        from RecoPixelVertexing.PixelTriplets.caHitTripletEDProducer_cfi import caHitTripletEDProducer as _caHitTripletEDProducer
        setattr(process, "initialStepHitTriplets"+postfix, _caHitTripletEDProducer.clone(
            doublets = "initialStepHitDoublets"+postfix,
            extraHitRPhitolerance = getattr(process, "initialStepHitQuadruplets"+postfix).extraHitRPhitolerance,
            SeedComparitorPSet = getattr(process, "initialStepHitQuadruplets"+postfix).SeedComparitorPSet.clone(),
            maxChi2 = dict(
                pt1    = 0.8, pt2    = 8,
                value1 = 100, value2 = 6,
            ),
            useBendingCorrection = True,
            CAThetaCut = 0.004,
            CAPhiCut = 0.07,
            CAHardPtCut = 0.3,
        ))

        getattr(process, "initialStepSeeds"+postfix).seedingHitSets = "initialStepHitTriplets"+postfix

        getattr(process, "InitialStep"+postfix).replace(
            getattr(process, "initialStepHitQuadruplets"+postfix),
            getattr(process, "initialStepHitTriplets"+postfix)
        )

    # Phase0-style combinations to pixelPairStep
    process.pixelPairStepSeedLayers.layerList = [
        'BPix2+BPix3',
        'BPix3+BPix4',
        'BPix2+BPix4',
        'BPix2+FPix1_pos', 'BPix2+FPix1_neg',
        'FPix1_pos+FPix2_pos', 'FPix1_neg+FPix2_neg',
        'FPix2_pos+FPix3_pos', 'FPix2_neg+FPix3_neg',
        'FPix1_pos+FPix3_pos', 'FPix1_neg+FPix3_neg',
    ]

    return process
