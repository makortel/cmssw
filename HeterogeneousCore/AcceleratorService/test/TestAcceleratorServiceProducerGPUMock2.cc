#include "HeterogeneousCore/AcceleratorService/interface/HeterogeneousEDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "tbb/concurrent_vector.h"

#include <chrono>
#include <future>
#include <random>
#include <thread>

class TestAcceleratorServiceProducerGPUMock2: public HeterogeneousEDProducer<heterogeneous::HeterogeneousDevices<
                                                                               heterogeneous::GPUMock,
                                                                               heterogeneous::CPU
                                                                               > > {
public:
  explicit TestAcceleratorServiceProducerGPUMock2(edm::ParameterSet const& iConfig);
  ~TestAcceleratorServiceProducerGPUMock2() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  using OutputType = HeterogeneousProductImpl<heterogeneous::CPUProduct<unsigned int>,
                                              heterogeneous::GPUMockProduct<unsigned int> >;

  std::string label_;
  edm::EDGetTokenT<HeterogeneousProduct> srcToken_;

  // hack for GPU mock
  tbb::concurrent_vector<std::future<void> > pendingFutures;

  const OutputType *input_ = nullptr;
  unsigned int eventId_ = 0;
  unsigned int streamId_ = 0;

  // simulating GPU memory
  unsigned int gpuOutput_;

  // output
  unsigned int output_;

  void acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  void launchCPU() override;
  void launchGPUMock(std::function<void()> callback);

  void produceCPU(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  void produceGPUMock(const HeterogeneousDeviceId& location, edm::Event& iEvent, const edm::EventSetup& iSetup) override;
};

TestAcceleratorServiceProducerGPUMock2::TestAcceleratorServiceProducerGPUMock2(edm::ParameterSet const& iConfig):
  label_(iConfig.getParameter<std::string>("@module_label"))
{
  auto srcTag = iConfig.getParameter<edm::InputTag>("src");
  if(!srcTag.label().empty()) {
    srcToken_ = consumesHeterogeneous(srcTag);
  }

  produces<HeterogeneousProduct>();
}

void TestAcceleratorServiceProducerGPUMock2::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag());
  descriptions.add("testAcceleratorServiceProducerGPUMock2", desc);
}

void TestAcceleratorServiceProducerGPUMock2::acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogPrint("TestAcceleratorServiceProducerGPUMock2") << label_ << " TestAcceleratorServiceProducerGPUMock2::acquire event " << iEvent.id().event() << " stream " << iEvent.streamID();

  input_ = nullptr;
  if(!srcToken_.isUninitialized()) {
    edm::Handle<HeterogeneousProduct> hin;
    iEvent.getByToken(srcToken_, hin);
    input_ = &(hin->get<OutputType>());
  }

  eventId_ = iEvent.id().event();
  streamId_ = iEvent.streamID();
}

void TestAcceleratorServiceProducerGPUMock2::launchCPU() {
  edm::LogPrint("TestAcceleratorServiceProducerGPUMock2") << " " << label_ << " TestAcceleratorServiceProducerGPUMock2::launchCPU begin event " << eventId_ << " stream " << streamId_;

  std::random_device r;
  std::mt19937 gen(r());
  auto dist = std::uniform_real_distribution<>(1.0, 3.0); 
  auto dur = dist(gen);
  edm::LogPrint("TestAcceleratorServiceProducerGPUMock2") << "  Task (CPU) for event " << eventId_ << " in stream " << streamId_ << " will take " << dur << " seconds";
  std::this_thread::sleep_for(std::chrono::seconds(1)*dur);

  auto input = input_ ? input_->getProduct<HeterogeneousDevice::kCPU>() : 0U;

  output_ = input + streamId_*100 + eventId_;

  edm::LogPrint("TestAcceleratorServiceProducerGPUMock2") << " " << label_ << " TestAcceleratorServiceProducerGPUMock2::launchCPU end event " << eventId_ << " stream " << streamId_;
}

void TestAcceleratorServiceProducerGPUMock2::launchGPUMock(std::function<void()> callback) {
  edm::LogPrint("TestAcceleratorServiceProducerGPUMock2") << " " << label_ << " TestAcceleratorServiceProducerGPUMock2::launchGPUMock begin event " << eventId_ << " stream " << streamId_;

  /// GPU work
  std::random_device r;
  std::mt19937 gen(r());
  auto dist = std::uniform_real_distribution<>(0.1, 1.0); 
  auto dur = dist(gen);
  edm::LogPrint("TestAcceleratorServiceProducerGPUMock2") << "  " << label_ << " Task (GPU) for event " << eventId_ << " in stream " << streamId_ << " will take " << dur << " seconds";

  auto input = input_ ? input_->getProduct<HeterogeneousDevice::kGPUMock>() : 0U;

  auto ret = std::async(std::launch::async,
                        [this, dur, input,
                         callback = std::move(callback)
                         ](){
                          std::this_thread::sleep_for(std::chrono::seconds(1)*dur);
                          gpuOutput_ = input + streamId_*100 + eventId_;
                          edm::LogPrint("TestAcceleratorServiceProducerGPUMock2") << "   " << label_ << " TestAcceleratorServiceProducerGPUMock2::launchGPUMock finished async for event " << eventId_ << " stream " << streamId_;
                          callback();
                        });
  pendingFutures.push_back(std::move(ret));

  edm::LogPrint("TestAcceleratorServiceProducerGPUMock2") << " " << label_ << " TestAcceleratorServiceProducerGPUMock2::launchGPUMock end event " << eventId_ << " stream " << streamId_;
}

void TestAcceleratorServiceProducerGPUMock2::produceCPU(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogPrint("TestAcceleratorServiceProducerGPUMock2") << label_ << " TestAcceleratorServiceProducerGPUMock2::produceCPU begin event " << iEvent.id().event() << " stream " << iEvent.streamID();

  iEvent.put(std::make_unique<HeterogeneousProduct>(OutputType(heterogeneous::cpuProduct(std::move(output_)))));

  edm::LogPrint("TestAcceleratorServiceProducerGPUMock2") << label_ << " TestAcceleratorServiceProducerGPUMock2::produceCPU end event " << iEvent.id().event() << " stream " << iEvent.streamID() << " result " << output_;
}

void TestAcceleratorServiceProducerGPUMock2::produceGPUMock(const HeterogeneousDeviceId& location, edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogPrint("TestAcceleratorServiceProducerGPUMock2") << label_ << " TestAcceleratorServiceProducerGPUMock2::produceGPUMock begin event " << iEvent.id().event() << " stream " << iEvent.streamID();

  iEvent.put(std::make_unique<HeterogeneousProduct>(OutputType(heterogeneous::gpuMockProduct(std::move(gpuOutput_)),
                                                               location,
                                                               [this](const unsigned int& src, unsigned int& dst) {
                                                                 edm::LogPrint("TestAcceleratorServiceProducerGPUMock2") << "  " << label_ << " Task (GPU) for event " << eventId_ << " in stream " << streamId_ << " copying to CPU";
                                                                 dst = src;
                                                               })));

  edm::LogPrint("TestAcceleratorServiceProducerGPUMock2") << label_ << " TestAcceleratorServiceProducerGPUMock2::produceGPUMock end event " << iEvent.id().event() << " stream " << iEvent.streamID() << " result " << gpuOutput_;
}

DEFINE_FWK_MODULE(TestAcceleratorServiceProducerGPUMock2);
