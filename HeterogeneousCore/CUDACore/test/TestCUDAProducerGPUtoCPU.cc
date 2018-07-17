#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "HeterogeneousCore/CUDACore/interface/CUDA.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"

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
  edm::EDGetTokenT<CUDA<float *>> srcToken_;
  cuda::memory::host::unique_ptr<float[]> buffer_;
};

TestCUDAProducerGPUtoCPU::TestCUDAProducerGPUtoCPU(const edm::ParameterSet& iConfig):
  label_(iConfig.getParameter<std::string>("@module_label")),
  srcToken_(consumes<CUDA<float *>>(iConfig.getParameter<edm::InputTag>("src"))),
  buffer_(cuda::memory::host::make_unique<float[]>(TestCUDAProducerGPUKernel::NUM_VALUES))
{
  produces<int>();
}

void TestCUDAProducerGPUtoCPU::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag())->setComment("Source for CUDA<float *>.");
  descriptions.addWithDefaultLabel(desc);
  descriptions.setComment("This EDProducer is part of the TestCUDAProducer* family. It models the GPU->CPU data transfer and formatting of the data to legacy data format. Produces int, to be compatible with TestCUDAProducerCPU.");
}

void TestCUDAProducerGPUtoCPU::acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  edm::LogPrint("TestCUDAProducerGPUtoCPU") << label_ << " TestCUDAProducerGPUtoCPU::acquire begin event " << iEvent.id().event() << " stream " << iEvent.streamID();

  edm::Handle<CUDA<float *>> hin;
  iEvent.getByToken(srcToken_, hin);
  auto ctx = CUDAScopedContext(*hin, std::move(waitingTaskHolder));
  const float *device = ctx.get(*hin);

  // Enqueue async copy, continue in produce once finished
  cuda::memory::async::copy(buffer_.get(), device, TestCUDAProducerGPUKernel::NUM_VALUES*sizeof(float), ctx.stream().id());

  edm::LogPrint("TestCUDAProducerGPUtoCPU") << label_ << " TestCUDAProducerGPUtoCPU::acquire end event " << iEvent.id().event() << " stream " << iEvent.streamID();
}

void TestCUDAProducerGPUtoCPU::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogPrint("TestCUDAProducerGPUtoCPU") << label_ << " TestCUDAProducerGPUtoCPU::produce begin event " << iEvent.id().event() << " stream " << iEvent.streamID();

  int counter = 0;
  for(int i=0; i<TestCUDAProducerGPUKernel::NUM_VALUES; ++i) {
    counter += buffer_[i];
  }

  iEvent.put(std::make_unique<int>(counter));

  edm::LogPrint("TestCUDAProducerGPUtoCPU") << label_ << " TestCUDAProducerGPUtoCPU::produce end event " << iEvent.id().event() << " stream " << iEvent.streamID() << " result " << counter;
}

DEFINE_FWK_MODULE(TestCUDAProducerGPUtoCPU);
