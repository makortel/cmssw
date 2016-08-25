from Configuration.StandardSequences.Eras import eras
from RecoPixelVertexing.PixelTriplets.pixelTripletHLTEDProducer_cfi import *
eras.trackingLowPU.toModify(pixelTripletHLTEDProducer, maxElement=100000)
eras.trackingPhase1PU70.toModify(pixelTripletHLTEDProducer, maxElement=0)
eras.trackingPhase2PU140.toModify(pixelTripletHLTEDProducer, maxElement=0)
