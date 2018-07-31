#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "HeterogeneousCore/CUDACore/interface/CUDAStreamEDProducer.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"
#include "HeterogeneousCore/CUDACore/interface/CUDA.h"

#include "TestCUDAProducerGPUKernel.h"

class TestCUDAProducerGPU: public CUDAStreamEDProducer<> {
public:
  explicit TestCUDAProducerGPU(const edm::ParameterSet& iConfig);
  ~TestCUDAProducerGPU() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginStreamCUDA(edm::StreamID id) override;

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup);
private:
  std::string label_;
  edm::EDGetTokenT<CUDA<float *>> srcToken_;
  std::unique_ptr<TestCUDAProducerGPUKernel> gpuAlgo_;
};

TestCUDAProducerGPU::TestCUDAProducerGPU(const edm::ParameterSet& iConfig):
  label_(iConfig.getParameter<std::string>("@module_label")),
  srcToken_(consumes<CUDA<float *>>(iConfig.getParameter<edm::InputTag>("src")))
{
  produces<CUDA<float *>>();
}

void TestCUDAProducerGPU::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag())->setComment("Source of CUDA<float *>.");
  descriptions.addWithDefaultLabel(desc);
  descriptions.setComment("This EDProducer is part of the TestCUDAProducer* family. It models a GPU algorithm this is not the first algorithm in the chain of the GPU EDProducers. Produces CUDA<float *>.");
}

void TestCUDAProducerGPU::beginStreamCUDA(edm::StreamID id) {
  // Allocate device memory via RAII
  gpuAlgo_ = std::make_unique<TestCUDAProducerGPUKernel>();
}

void TestCUDAProducerGPU::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogPrint("TestCUDAProducerGPU") << label_ << " TestCUDAProducerGPU::produce begin event " << iEvent.id().event() << " stream " << iEvent.streamID();

  edm::Handle<CUDA<float *> > hin;
  iEvent.getByToken(srcToken_, hin);
  auto ctx = CUDAScopedContext(*hin);
  const float *input = ctx.get(*hin);

  iEvent.put(ctx.wrap(gpuAlgo_->runAlgo(label_, input, ctx.stream())));

  edm::LogPrint("TestCUDAProducerGPU") << label_ << " TestCUDAProducerGPU::produce end event " << iEvent.id().event() << " stream " << iEvent.streamID();
}

DEFINE_FWK_MODULE(TestCUDAProducerGPU);
