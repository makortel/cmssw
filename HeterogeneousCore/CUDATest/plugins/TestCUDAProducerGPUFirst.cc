#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "HeterogeneousCore/CUDACore/interface/CUDAStreamEDProducer.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"
#include "HeterogeneousCore/CUDACore/interface/CUDA.h"

#include "TestCUDAProducerGPUKernel.h"

class TestCUDAProducerGPUFirst: public CUDAStreamEDProducer<> {
public:
  explicit TestCUDAProducerGPUFirst(const edm::ParameterSet& iConfig);
  ~TestCUDAProducerGPUFirst() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginStreamCUDA(edm::StreamID id) override;

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup);
private:
  std::string label_;
  std::unique_ptr<TestCUDAProducerGPUKernel> gpuAlgo_;
};

TestCUDAProducerGPUFirst::TestCUDAProducerGPUFirst(const edm::ParameterSet& iConfig):
  label_(iConfig.getParameter<std::string>("@module_label"))
{
  produces<CUDA<float *>>();
}

void TestCUDAProducerGPUFirst::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.addWithDefaultLabel(desc);
  descriptions.setComment("This EDProducer is part of the TestCUDAProducer* family. It models a GPU algorithm this the first algorithm in the chain of the GPU EDProducers. Produces CUDA<float *>.");
}

void TestCUDAProducerGPUFirst::beginStreamCUDA(edm::StreamID id) {
  // Allocate device memory via RAII
  gpuAlgo_ = std::make_unique<TestCUDAProducerGPUKernel>();
}

void TestCUDAProducerGPUFirst::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogPrint("TestCUDAProducerGPUFirst") << label_ << " TestCUDAProducerGPUFirst::produce begin event " << iEvent.id().event() << " stream " << iEvent.streamID();

  auto ctx = CUDAScopedContext(iEvent.streamID());

  float *output = gpuAlgo_->runAlgo(label_, ctx.stream());
  iEvent.put(ctx.wrap(output));

  edm::LogPrint("TestCUDAProducerGPUFirst") << label_ << " TestCUDAProducerGPUFirst::produce end event " << iEvent.id().event() << " stream " << iEvent.streamID();
}

DEFINE_FWK_MODULE(TestCUDAProducerGPUFirst);
