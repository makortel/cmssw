#include "HeterogeneousCore/AcceleratorService/interface/HeterogeneousEDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"

#include "TestAcceleratorServiceProducerGPUHelpers.h"

#include <chrono>
#include <random>
#include <thread>

#include <cuda.h>
#include <cuda_runtime.h>


class TestAcceleratorServiceProducerGPU2: public HeterogeneousEDProducer<heterogeneous::HeterogeneousDevices <
                                                                           heterogeneous::GPUCuda,
                                                                           heterogeneous::CPU
                                                                           > > {
public:
  explicit TestAcceleratorServiceProducerGPU2(edm::ParameterSet const& iConfig);
  ~TestAcceleratorServiceProducerGPU2() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  using OutputType = HeterogeneousProductImpl<heterogeneous::CPUProduct<unsigned int>,
                                              heterogeneous::GPUCudaProduct<TestAcceleratorServiceProducerGPUTask::ResultTypeRaw>>;

  void acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  void launchCPU() override;
  void launchGPUCuda(CallbackType callback) override;

  void produceCPU(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  void produceGPUCuda(const HeterogeneousDeviceId& location, edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  std::string label_;
  edm::EDGetTokenT<HeterogeneousProduct> srcToken_;

  // input
  const OutputType *input_ = nullptr;
  unsigned int eventId_ = 0;
  unsigned int streamId_ = 0;

  // GPU stuff
  std::unique_ptr<TestAcceleratorServiceProducerGPUTask> gpuAlgo_;
  TestAcceleratorServiceProducerGPUTask::ResultType gpuOutput_;

    // output
  unsigned int output_;
};

TestAcceleratorServiceProducerGPU2::TestAcceleratorServiceProducerGPU2(edm::ParameterSet const& iConfig):
  label_(iConfig.getParameter<std::string>("@module_label"))
{
  auto srcTag = iConfig.getParameter<edm::InputTag>("src");
  if(!srcTag.label().empty()) {
    srcToken_ = consumes<HeterogeneousProduct>(srcTag);
  }

  edm::Service<CUDAService> cudaService;
  if(cudaService->enabled()) {
    gpuAlgo_ = std::make_unique<TestAcceleratorServiceProducerGPUTask>();
  }

  produces<HeterogeneousProduct>();
}

void TestAcceleratorServiceProducerGPU2::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag());
  descriptions.add("testAcceleratorServiceProducerGPU2", desc);
}

void TestAcceleratorServiceProducerGPU2::acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogPrint("TestAcceleratorServiceProducerGPU2") << label_ << " TestAcceleratorServiceProducerGPU2::acquire event " << iEvent.id().event() << " stream " << iEvent.streamID();

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

  // I don't like this call but i don't have good ideas for how to get around that...
  schedule(input_);
}

void TestAcceleratorServiceProducerGPU2::launchCPU() {
  edm::LogPrint("TestAcceleratorServiceProducerGPU2") << " " << label_ << " TestAcceleratorServiceProducerGPU2::launchCPU begin event " << eventId_ << " stream " << streamId_;

  std::random_device r;
  std::mt19937 gen(r());
  auto dist = std::uniform_real_distribution<>(1.0, 3.0); 
  auto dur = dist(gen);
  edm::LogPrint("TestAcceleratorServiceProducerGPU2") << "  Task (CPU) for event " << eventId_ << " in stream " << streamId_ << " will take " << dur << " seconds";
  std::this_thread::sleep_for(std::chrono::seconds(1)*dur);

  auto input = input_ ? input_->getProduct<HeterogeneousDevice::kCPU>() : 0U;

  output_ = input + streamId_*100 + eventId_;

  edm::LogPrint("TestAcceleratorServiceProducerGPU2") << " " << label_ << " TestAcceleratorServiceProducerGPU2::launchCPU end event " << eventId_ << " stream " << streamId_;
}

void TestAcceleratorServiceProducerGPU2::launchGPUCuda(CallbackType callback) {
  edm::Service<CUDAService> cs;
  edm::LogPrint("TestAcceleratorServiceProducerGPU2") << " " << label_ << " TestAcceleratorServiceProducerGPU2::launchGPUCuda begin event " << eventId_ << " stream " << streamId_ << " device " << cs->getCurrentDevice();

  gpuOutput_ = gpuAlgo_->runAlgo(label_, 0, input_ ? input_->getProduct<HeterogeneousDevice::kGPUCuda>() : std::make_pair(nullptr, nullptr), callback);

  edm::LogPrint("TestAcceleratorServiceProducerGPU2") << " " << label_ << " TestAcceleratorServiceProducerGPU2::launchGPUCuda end event " << eventId_ << " stream " << streamId_ << " device " << cs->getCurrentDevice();
}

void TestAcceleratorServiceProducerGPU2::produceCPU(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogPrint("TestAcceleratorServiceProducerGPU2") << label_ << " TestAcceleratorServiceProducerGPU2::produceCPU begin event " << iEvent.id().event() << " stream " << iEvent.streamID();

  iEvent.put(std::make_unique<HeterogeneousProduct>(OutputType(heterogeneous::cpuProduct(std::move(output_)))));

  edm::LogPrint("TestAcceleratorServiceProducerGPU2") << label_ << " TestAcceleratorServiceProducerGPU2::produceCPU end event " << iEvent.id().event() << " stream " << iEvent.streamID() << " result " << output_;
}

void TestAcceleratorServiceProducerGPU2::produceGPUCuda(const HeterogeneousDeviceId& location, edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Service<CUDAService> cs;
  edm::LogPrint("TestAcceleratorServiceProducerGPU2") << label_ << " TestAcceleratorServiceProducerGPU2::produceGPUCuda begin event " << iEvent.id().event() << " stream " << iEvent.streamID() << " device " << cs->getCurrentDevice();

  gpuAlgo_->release(label_);
  iEvent.put(std::make_unique<HeterogeneousProduct>(OutputType(heterogeneous::gpuCudaProduct(std::make_pair(gpuOutput_.first.get(), gpuOutput_.second.get())),
                                                               location,
                                                               [this, eventId=iEvent.id().event(), streamId=iEvent.streamID()](const TestAcceleratorServiceProducerGPUTask::ResultTypeRaw& src, unsigned int& dst) {
                                                                 edm::LogPrint("TestAcceleratorServiceProducerGPU2") << "  " << label_ << " Copying from GPU to CPU for event " << eventId << " in stream " << streamId;
                                                                 dst = TestAcceleratorServiceProducerGPUTask::getResult(src);
                                                               })));

  edm::LogPrint("TestAcceleratorServiceProducerGPU2") << label_ << " TestAcceleratorServiceProducerGPU2::produceGPUCuda end event " << iEvent.id().event() << " stream " << iEvent.streamID() << " device " << cs->getCurrentDevice();
}

DEFINE_FWK_MODULE(TestAcceleratorServiceProducerGPU2);
