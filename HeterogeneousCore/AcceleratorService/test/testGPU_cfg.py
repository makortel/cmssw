import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.options = cms.untracked.PSet(
#    numberOfThreads = cms.untracked.uint32(4),
    numberOfStreams = cms.untracked.uint32(0)
)


#process.Tracer = cms.Service("Tracer")
process.AcceleratorService = cms.Service("AcceleratorService")
process.CudaService = cms.Service("CudaService")
process.prod1 = cms.EDProducer('TestAcceleratorServiceProducerGPU')
process.prod2 = cms.EDProducer('TestAcceleratorServiceProducerGPU',
    src = cms.InputTag("prod1"),
)
process.ana2 = cms.EDAnalyzer("TestAcceleratorServiceAnalyzer",
    src = cms.InputTag("prod2")
)

process.prod3 = cms.EDProducer('TestAcceleratorServiceProducerGPU',
    src = cms.InputTag("prod1"),
)
process.ana3 = cms.EDAnalyzer("TestAcceleratorServiceAnalyzer",
    src = cms.InputTag("prod3")
)


process.t = cms.Task(process.prod1, process.prod2, process.prod3)

process.p = cms.Path(process.ana2+process.ana3)
process.p.associate(process.t)
