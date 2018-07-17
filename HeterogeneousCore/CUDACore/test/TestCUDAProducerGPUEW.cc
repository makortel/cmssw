#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "HeterogeneousCore/CUDACore/interface/CUDAStreamEDProducer.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"
#include "HeterogeneousCore/CUDACore/interface/CUDA.h"

#include "TestCUDAProducerGPUKernel.h"

class TestCUDAProducerGPUEW: public CUDAStreamEDProducer<edm::ExternalWork> {
public:
  explicit TestCUDAProducerGPUEW(const edm::ParameterSet& iConfig);
  ~TestCUDAProducerGPUEW() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginStreamCUDA(edm::StreamID id) override;

  void acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
private:
  std::string label_;
  edm::EDGetTokenT<CUDA<float *>> srcToken_;
  std::unique_ptr<TestCUDAProducerGPUKernel> gpuAlgo_;
  float *devicePtr_ = nullptr;
  float hostData_ = 0.f;
};

TestCUDAProducerGPUEW::TestCUDAProducerGPUEW(const edm::ParameterSet& iConfig):
  label_(iConfig.getParameter<std::string>("@module_label")),
  srcToken_(consumes<CUDA<float *>>(iConfig.getParameter<edm::InputTag>("src")))
{
  produces<CUDA<float *>>();
}

void TestCUDAProducerGPUEW::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag());
  descriptions.addWithDefaultLabel(desc);
}

void TestCUDAProducerGPUEW::beginStreamCUDA(edm::StreamID id) {
  // Allocate device memory via RAII
  gpuAlgo_ = std::make_unique<TestCUDAProducerGPUKernel>();
}

void TestCUDAProducerGPUEW::acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  edm::LogPrint("TestCUDAProducerGPUEW") << label_ << " TestCUDAProducerGPUEW::acquire begin event " << iEvent.id().event() << " stream " << iEvent.streamID();

  edm::Handle<CUDA<float *> > hin;
  iEvent.getByToken(srcToken_, hin);
  auto ctx = CUDAScopedContext(*hin, std::move(waitingTaskHolder));
  const float *input = ctx.get(*hin);

  devicePtr_ = gpuAlgo_->runAlgo(label_, input, ctx.stream());
  // Mimick the need to transfer some of the GPU data back to CPU to
  // be used for something within this module, or to be put in the
  // event.
  cuda::memory::async::copy(&hostData_, devicePtr_+10, sizeof(float), ctx.stream().id());

  edm::LogPrint("TestCUDAProducerGPUEW") << label_ << " TestCUDAProducerGPUEW::acquire end event " << iEvent.id().event() << " stream " << iEvent.streamID();
}

void TestCUDAProducerGPUEW::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogPrint("TestCUDAProducerGPUEW") << label_ << " TestCUDAProducerGPUEW::produce begin event " << iEvent.id().event() << " stream " << iEvent.streamID() << " 10th element " << hostData_; 

  // It feels a bit stupid to read the input again here, but for
  // anything else we'd need to somehow transfer the device+stream
  // information from acquire.
  edm::Handle<CUDA<float *> > hin;
  iEvent.getByToken(srcToken_, hin);
  auto ctx = CUDAScopedContext(*hin);

  iEvent.put(ctx.wrap(devicePtr_));
  devicePtr_ = nullptr;

  edm::LogPrint("TestCUDAProducerGPUEW") << label_ << " TestCUDAProducerGPUEW::produce end event " << iEvent.id().event() << " stream " << iEvent.streamID();
}

DEFINE_FWK_MODULE(TestCUDAProducerGPUEW);
