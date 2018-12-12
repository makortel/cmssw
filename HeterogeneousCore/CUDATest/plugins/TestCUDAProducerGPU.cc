#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "CUDADataFormats/Common/interface/CUDA.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"
#include "HeterogeneousCore/CUDATest/interface/CUDAThing.h"

#include "TestCUDAProducerGPUKernel.h"

class TestCUDAProducerGPU: public edm::global::EDProducer<> {
public:
  explicit TestCUDAProducerGPU(const edm::ParameterSet& iConfig);
  ~TestCUDAProducerGPU() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;
private:
  std::string label_;
  edm::EDGetTokenT<CUDA<CUDAThing>> srcToken_;
  TestCUDAProducerGPUKernel gpuAlgo_;
};

TestCUDAProducerGPU::TestCUDAProducerGPU(const edm::ParameterSet& iConfig):
  label_(iConfig.getParameter<std::string>("@module_label")),
  srcToken_(consumes<CUDA<CUDAThing>>(iConfig.getParameter<edm::InputTag>("src")))
{
  produces<CUDA<CUDAThing>>();
}

void TestCUDAProducerGPU::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag())->setComment("Source of CUDA<CUDAThing>.");
  descriptions.addWithDefaultLabel(desc);
  descriptions.setComment("This EDProducer is part of the TestCUDAProducer* family. It models a GPU algorithm this is not the first algorithm in the chain of the GPU EDProducers. Produces CUDA<CUDAThing>.");
}

void TestCUDAProducerGPU::produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::LogPrint("TestCUDAProducerGPU") << label_ << " TestCUDAProducerGPU::produce begin event " << iEvent.id().event() << " stream " << iEvent.streamID();

  edm::Handle<CUDA<CUDAThing>> hin;
  iEvent.getByToken(srcToken_, hin);
  auto ctx = CUDAScopedContext(*hin);
  const CUDAThing& input = ctx.get(*hin);

  iEvent.put(ctx.wrap(CUDAThing(gpuAlgo_.runAlgo(label_, input.get(), ctx.stream()))));

  edm::LogPrint("TestCUDAProducerGPU") << label_ << " TestCUDAProducerGPU::produce end event " << iEvent.id().event() << " stream " << iEvent.streamID();
}

DEFINE_FWK_MODULE(TestCUDAProducerGPU);
