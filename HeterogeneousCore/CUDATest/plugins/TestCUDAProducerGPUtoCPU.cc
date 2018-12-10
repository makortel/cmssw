#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "HeterogeneousCore/CUDACore/interface/CUDA.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDATest/interface/CUDAThing.h"

#include "TestCUDAProducerGPUKernel.h"

class TestCUDAProducerGPUtoCPU: public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit TestCUDAProducerGPUtoCPU(const edm::ParameterSet& iConfig);
  ~TestCUDAProducerGPUtoCPU() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
private:
  std::string label_;
  edm::EDGetTokenT<CUDA<CUDAThing>> srcToken_;
  edm::cuda::host::unique_ptr<float[]> buffer_;
};

TestCUDAProducerGPUtoCPU::TestCUDAProducerGPUtoCPU(const edm::ParameterSet& iConfig):
  label_(iConfig.getParameter<std::string>("@module_label")),
  srcToken_(consumes<CUDA<CUDAThing>>(iConfig.getParameter<edm::InputTag>("src")))
{
  produces<int>();
}

void TestCUDAProducerGPUtoCPU::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag())->setComment("Source for CUDA<CUDAThing>.");
  descriptions.addWithDefaultLabel(desc);
  descriptions.setComment("This EDProducer is part of the TestCUDAProducer* family. It models the GPU->CPU data transfer and formatting of the data to legacy data format. Produces int, to be compatible with TestCUDAProducerCPU.");
}

void TestCUDAProducerGPUtoCPU::acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  edm::LogPrint("TestCUDAProducerGPUtoCPU") << label_ << " TestCUDAProducerGPUtoCPU::acquire begin event " << iEvent.id().event() << " stream " << iEvent.streamID();

  edm::Handle<CUDA<CUDAThing>> hin;
  iEvent.getByToken(srcToken_, hin);
  auto ctx = CUDAScopedContext(*hin, std::move(waitingTaskHolder));
  const CUDAThing& device = ctx.get(*hin);

  edm::Service<CUDAService> cs;
  buffer_ = cs->make_host_unique<float[]>(TestCUDAProducerGPUKernel::NUM_VALUES, ctx.stream());
  // Enqueue async copy, continue in produce once finished
  cuda::memory::async::copy(buffer_.get(), device.get(), TestCUDAProducerGPUKernel::NUM_VALUES*sizeof(float), ctx.stream().id());

  edm::LogPrint("TestCUDAProducerGPUtoCPU") << label_ << " TestCUDAProducerGPUtoCPU::acquire end event " << iEvent.id().event() << " stream " << iEvent.streamID();
}

void TestCUDAProducerGPUtoCPU::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogPrint("TestCUDAProducerGPUtoCPU") << label_ << " TestCUDAProducerGPUtoCPU::produce begin event " << iEvent.id().event() << " stream " << iEvent.streamID();

  int counter = 0;
  for(int i=0; i<TestCUDAProducerGPUKernel::NUM_VALUES; ++i) {
    counter += buffer_[i];
  }
  buffer_.reset(); // not so nice, but no way around?

  iEvent.put(std::make_unique<int>(counter));

  edm::LogPrint("TestCUDAProducerGPUtoCPU") << label_ << " TestCUDAProducerGPUtoCPU::produce end event " << iEvent.id().event() << " stream " << iEvent.streamID() << " result " << counter;
}

DEFINE_FWK_MODULE(TestCUDAProducerGPUtoCPU);
