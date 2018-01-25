#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HeterogeneousCore/AcceleratorService/interface/AcceleratorService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"

#include "tbb/concurrent_vector.h"

#include <chrono>
#include <future>
#include <random>
#include <thread>

namespace {
  // hack for GPU mock
  tbb::concurrent_vector<std::future<void> > pendingFutures;

  using OutputType = HeterogeneousProduct<unsigned int, unsigned int>;

  class TestTask: public AcceleratorTask<accelerator::CPU, accelerator::GPUMock> {
  public:
    TestTask(const OutputType *input, unsigned int eventId, unsigned int streamId):
      input_(input), eventId_(eventId), streamId_(streamId) {}
    ~TestTask() override = default;

    accelerator::Capabilities preferredDevice() const override {
      if(input_ == nullptr) {
        // Without input decide randomly whether to run on GPU or CPU to simulate scheduler decisions
        std::random_device r;
        std::mt19937 gen(r());
        auto dist1 = std::uniform_int_distribution<>(0, 1); // simulate the scheduler decision
        return dist1(gen) == 0 ? accelerator::Capabilities::kGPUMock : accelerator::Capabilities::kCPU;
      }
      else if(input_->isProductOn(HeterogeneousLocation::kGPU)) {
        return accelerator::Capabilities::kGPUMock;
      }
      else {
        return accelerator::Capabilities::kCPU;
      }
    }

    void run_CPU() override {
      std::random_device r;
      std::mt19937 gen(r());
      auto dist = std::uniform_real_distribution<>(1.0, 3.0); 
      auto dur = dist(gen);
      edm::LogPrint("Foo") << "   Task (CPU) for event " << eventId_ << " in stream " << streamId_ << " will take " << dur << " seconds";
      std::this_thread::sleep_for(std::chrono::seconds(1)*dur);

      auto input = input_ ? input_->getCPUProduct() : 0U;

      output_ = input + streamId_*100 + eventId_;
    }

    void run_GPUMock(std::function<void()> callback) override {
      std::random_device r;
      std::mt19937 gen(r());
      auto dist = std::uniform_real_distribution<>(0.1, 1.0); 
      auto dur = dist(gen);
      edm::LogPrint("Foo") << "   Task (GPU) for event " << eventId_ << " in stream " << streamId_ << " will take " << dur << " seconds";
      ranOnGPU_ = true;
      auto input = input_ ? input_->getGPUProduct() : 0U;

      auto ret = std::async(std::launch::async,
                            [this, dur, input,
                             callback = std::move(callback)
                             ](){
                              std::this_thread::sleep_for(std::chrono::seconds(1)*dur);
                              gpuOutput_ = input + streamId_*100 + eventId_;
                              callback();
                            });
      pendingFutures.push_back(std::move(ret));
    }

    auto makeTransfer() const {
      return [this](const unsigned int& src, unsigned int& dst) {
        edm::LogPrint("Foo") << "   Task (GPU) for event " << eventId_ << " in stream " << streamId_ << " copying to CPU";
        dst = src;
      };
    }

    bool ranOnGPU() const { return ranOnGPU_; }
    unsigned int getOutput() const { return output_; }
    unsigned int getGPUOutput() const { return gpuOutput_; }

  private:
    // input
    const OutputType *input_;
    unsigned int eventId_;
    unsigned int streamId_;

    bool ranOnGPU_ = false;

    // simulating GPU memory
    unsigned int gpuOutput_;

    // output
    unsigned int output_;
  };
}

class TestAcceleratorServiceProducerGPUMock: public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit TestAcceleratorServiceProducerGPUMock(edm::ParameterSet const& iConfig);
  ~TestAcceleratorServiceProducerGPUMock() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  std::string label_;
  AcceleratorService::Token accToken_;

  edm::EDGetTokenT<OutputType> srcToken_;

  // to mimic external task worker interface
  void acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTask) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
};


TestAcceleratorServiceProducerGPUMock::TestAcceleratorServiceProducerGPUMock(const edm::ParameterSet& iConfig):
  label_(iConfig.getParameter<std::string>("@module_label")),
  accToken_(edm::Service<AcceleratorService>()->book())
{
  auto srcTag = iConfig.getParameter<edm::InputTag>("src");
  if(!srcTag.label().empty()) {
    srcToken_ = consumes<OutputType>(srcTag);
  }

  produces<OutputType>();
}

void TestAcceleratorServiceProducerGPUMock::acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  const OutputType *input = nullptr;
  if(!srcToken_.isUninitialized()) {
    edm::Handle<OutputType> hin;
    iEvent.getByToken(srcToken_, hin);
    input = hin.product();
  }

  edm::LogPrint("Foo") << "TestAcceleratorServiceProducerGPUMock::acquire begin event " << iEvent.id().event() << " stream " << iEvent.streamID() << " label " << label_ << " input " << input;
  edm::Service<AcceleratorService> acc;
  acc->async(accToken_, iEvent.streamID(), std::make_unique<::TestTask>(input, iEvent.id().event(), iEvent.streamID()), std::move(waitingTaskHolder));
  edm::LogPrint("Foo") << "TestAcceleratorServiceProducerGPUMock::acquire end event " << iEvent.id().event() << " stream " << iEvent.streamID() << " label " << label_;
}

void TestAcceleratorServiceProducerGPUMock::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogPrint("Foo") << "TestAcceleratorServiceProducerGPUMock::produce begin event " << iEvent.id().event() << " stream " << iEvent.streamID() << " label " << label_;
  edm::Service<AcceleratorService> acc;
  const auto& task = dynamic_cast<const ::TestTask&>(acc->getTask(accToken_, iEvent.streamID()));
  std::unique_ptr<OutputType> ret;
  unsigned int value = 0;
  if(task.ranOnGPU()) {
    ret = std::make_unique<OutputType>(task.getGPUOutput(), task.makeTransfer());
    value = ret->getGPUProduct();
  }
  else {
    ret = std::make_unique<OutputType>(task.getOutput());
    value = ret->getCPUProduct();
  }

  edm::LogPrint("Foo") << "TestAcceleratorServiceProducerGPUMock::produce end event " << iEvent.id().event() << " stream " << iEvent.streamID() << " label " << label_ << " result " << value;
  iEvent.put(std::move(ret));
}

void TestAcceleratorServiceProducerGPUMock::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag());
  descriptions.add("testAcceleratorServiceProducerGPUMock", desc);
}

DEFINE_FWK_MODULE(TestAcceleratorServiceProducerGPUMock);
