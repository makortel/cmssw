#include "HeterogeneousCore/Producer/interface/HeterogeneousEDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"

#include "TestHeterogeneousEDProducerGPUHelpers.h"

#include <chrono>
#include <random>
#include <thread>

#include <cuda.h>
#include <cuda_runtime.h>


class TestHeterogeneousEDProducerGPU: public HeterogeneousEDProducer<heterogeneous::HeterogeneousDevices <
                                                                       heterogeneous::GPUCuda,
                                                                       heterogeneous::CPU
                                                                       > > {
public:
  explicit TestHeterogeneousEDProducerGPU(edm::ParameterSet const& iConfig);
  ~TestHeterogeneousEDProducerGPU() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  using OutputType = HeterogeneousProductImpl<heterogeneous::CPUProduct<unsigned int>,
                                              heterogeneous::GPUCudaProduct<TestHeterogeneousEDProducerGPUTask::ResultTypeRaw>>;

  void acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  void launchCPU() override;
  void launchGPUCuda(CallbackType callback) override;

  void produceCPU(edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup) override;
  void produceGPUCuda(edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup) override;

  std::string label_;
  edm::EDGetTokenT<HeterogeneousProduct> srcToken_;

  // input
  const OutputType *input_ = nullptr;
  unsigned int eventId_ = 0;
  unsigned int streamId_ = 0;

  // GPU stuff
  std::unique_ptr<TestHeterogeneousEDProducerGPUTask> gpuAlgo_;
  TestHeterogeneousEDProducerGPUTask::ResultType gpuOutput_;

    // output
  unsigned int output_;
};

TestHeterogeneousEDProducerGPU::TestHeterogeneousEDProducerGPU(edm::ParameterSet const& iConfig):
  label_(iConfig.getParameter<std::string>("@module_label"))
{
  auto srcTag = iConfig.getParameter<edm::InputTag>("src");
  if(!srcTag.label().empty()) {
    srcToken_ = consumesHeterogeneous(srcTag);
  }

  edm::Service<CUDAService> cudaService;
  if(cudaService->enabled()) {
    gpuAlgo_ = std::make_unique<TestHeterogeneousEDProducerGPUTask>();
  }

  produces<HeterogeneousProduct>();
}

void TestHeterogeneousEDProducerGPU::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag());
  descriptions.add("testHeterogeneousEDProducerGPU2", desc);
}

void TestHeterogeneousEDProducerGPU::acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogPrint("TestHeterogeneousEDProducerGPU") << label_ << " TestHeterogeneousEDProducerGPU::acquire event " << iEvent.id().event() << " stream " << iEvent.streamID();

  gpuOutput_.first.reset();
  gpuOutput_.second.reset();

  input_ = nullptr;
  if(!srcToken_.isUninitialized()) {
    edm::Handle<HeterogeneousProduct> hin;
    iEvent.getByToken(srcToken_, hin);
    input_ = &(hin->get<OutputType>());
  }

  eventId_ = iEvent.id().event();
  streamId_ = iEvent.streamID();
}

void TestHeterogeneousEDProducerGPU::launchCPU() {
  edm::LogPrint("TestHeterogeneousEDProducerGPU") << " " << label_ << " TestHeterogeneousEDProducerGPU::launchCPU begin event " << eventId_ << " stream " << streamId_;

  std::random_device r;
  std::mt19937 gen(r());
  auto dist = std::uniform_real_distribution<>(1.0, 3.0); 
  auto dur = dist(gen);
  edm::LogPrint("TestHeterogeneousEDProducerGPU") << "  Task (CPU) for event " << eventId_ << " in stream " << streamId_ << " will take " << dur << " seconds";
  std::this_thread::sleep_for(std::chrono::seconds(1)*dur);

  auto input = input_ ? input_->getProduct<HeterogeneousDevice::kCPU>() : 0U;

  output_ = input + streamId_*100 + eventId_;

  edm::LogPrint("TestHeterogeneousEDProducerGPU") << " " << label_ << " TestHeterogeneousEDProducerGPU::launchCPU end event " << eventId_ << " stream " << streamId_;
}

void TestHeterogeneousEDProducerGPU::launchGPUCuda(CallbackType callback) {
  edm::Service<CUDAService> cs;
  edm::LogPrint("TestHeterogeneousEDProducerGPU") << " " << label_ << " TestHeterogeneousEDProducerGPU::launchGPUCuda begin event " << eventId_ << " stream " << streamId_ << " device " << cs->getCurrentDevice();

  gpuOutput_ = gpuAlgo_->runAlgo(label_, 0, input_ ? input_->getProduct<HeterogeneousDevice::kGPUCuda>() : std::make_pair(nullptr, nullptr), callback);

  edm::LogPrint("TestHeterogeneousEDProducerGPU") << " " << label_ << " TestHeterogeneousEDProducerGPU::launchGPUCuda end event " << eventId_ << " stream " << streamId_ << " device " << cs->getCurrentDevice();
}

void TestHeterogeneousEDProducerGPU::produceCPU(edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup) {
  edm::LogPrint("TestHeterogeneousEDProducerGPU") << label_ << " TestHeterogeneousEDProducerGPU::produceCPU begin event " << iEvent.id().event() << " stream " << iEvent.streamID();

  iEvent.put<OutputType>(std::make_unique<unsigned int>(output_));

  edm::LogPrint("TestHeterogeneousEDProducerGPU") << label_ << " TestHeterogeneousEDProducerGPU::produceCPU end event " << iEvent.id().event() << " stream " << iEvent.streamID() << " result " << output_;
}

void TestHeterogeneousEDProducerGPU::produceGPUCuda(edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup) {
  edm::Service<CUDAService> cs;
  edm::LogPrint("TestHeterogeneousEDProducerGPU") << label_ << " TestHeterogeneousEDProducerGPU::produceGPUCuda begin event " << iEvent.id().event() << " stream " << iEvent.streamID() << " device " << cs->getCurrentDevice();

  gpuAlgo_->release(label_);
  iEvent.put<OutputType>(std::make_unique<TestHeterogeneousEDProducerGPUTask::ResultTypeRaw>(gpuOutput_.first.get(), gpuOutput_.second.get()),
                         [this, eventId=iEvent.event().id().event(), streamId=iEvent.event().streamID()](const TestHeterogeneousEDProducerGPUTask::ResultTypeRaw& src, unsigned int& dst) {
                           edm::LogPrint("TestHeterogeneousEDProducerGPU") << "  " << label_ << " Copying from GPU to CPU for event " << eventId << " in stream " << streamId;
                           dst = TestHeterogeneousEDProducerGPUTask::getResult(src);
                         });

  edm::LogPrint("TestHeterogeneousEDProducerGPU") << label_ << " TestHeterogeneousEDProducerGPU::produceGPUCuda end event " << iEvent.id().event() << " stream " << iEvent.streamID() << " device " << cs->getCurrentDevice();
}

DEFINE_FWK_MODULE(TestHeterogeneousEDProducerGPU);
