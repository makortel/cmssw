#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/AcceleratorService/interface/AcceleratorService.h"
#include "HeterogeneousCore/CudaService/interface/CudaService.h"

#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"

#include "TestAcceleratorServiceProducerGPUHelpers.h"

#include <chrono>
#include <future>
#include <random>
#include <thread>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {
  using OutputType = HeterogeneousProduct<unsigned int, TestAcceleratorServiceProducerGPUTask::ResultTypeRaw>;

  class TestAlgo {
  public:
    TestAlgo() {
      edm::Service<CudaService> cudaService;
      if(cudaService->enabled()) {
        gpuAlgo_ = std::make_unique<TestAcceleratorServiceProducerGPUTask>();
      }
    }
    ~TestAlgo() = default;

    void setInput(const OutputType *input, unsigned int eventId, unsigned int streamId) {
      input_ = input;
      eventId_ = eventId;
      streamId_ = streamId;
    }
    
    void runCPU() {
      std::random_device r;
      std::mt19937 gen(r());
      auto dist = std::uniform_real_distribution<>(1.0, 3.0); 
      auto dur = dist(gen);
      edm::LogPrint("Foo") << "   Task (CPU) for event " << eventId_ << " in stream " << streamId_ << " will take " << dur << " seconds";
      std::this_thread::sleep_for(std::chrono::seconds(1)*dur);

      auto input = input_ ? input_->getCPUProduct() : 0U;

      output_ = input + streamId_*100 + eventId_;
    }

    void runGPUCuda(std::function<void()> callback) {
      edm::LogPrint("Foo") << "   Task (GPU) for event " << eventId_ << " in stream " << streamId_ << " running on GPU asynchronously";
      gpuOutput_ = gpuAlgo_->runAlgo(0, input_ ? input_->getGPUProduct() : nullptr, [callback,this](){
          edm::LogPrint("Foo") << "    GPU kernel finished (in callback)";
          ranOnGPU_ = true;
          callback();
        });
      edm::LogPrint("Foo") << "   Task (GPU) for event " << eventId_ << " in stream " << streamId_ << " launched";
    }

    auto makeTransfer() const {
      return [this](const TestAcceleratorServiceProducerGPUTask::ResultTypeRaw& src, unsigned int& dst) {
        edm::LogPrint("Foo") << "   Task (GPU) for event " << eventId_ << " in stream " << streamId_ << " copying to CPU";
        dst = gpuAlgo_->getResult(src);
        edm::LogPrint("Foo") << "    GPU result " << dst;
      };
    }

    bool ranOnGPU() const { return ranOnGPU_; }
    unsigned int getOutput() const { return output_; }
    const TestAcceleratorServiceProducerGPUTask::ResultTypeRaw getGPUOutput() const {
      gpuAlgo_->release();
      return gpuOutput_.get();
    }

  private:
    // input
    const OutputType *input_ = nullptr;
    unsigned int eventId_ = 0;
    unsigned int streamId_ = 0;

    // GPU stuff
    std::unique_ptr<TestAcceleratorServiceProducerGPUTask> gpuAlgo_;
    TestAcceleratorServiceProducerGPUTask::ResultType gpuOutput_;
    bool ranOnGPU_ = false;

    // output
    unsigned int output_;
  };
}

class TestAcceleratorServiceProducerGPU: public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit TestAcceleratorServiceProducerGPU(edm::ParameterSet const& iConfig);
  ~TestAcceleratorServiceProducerGPU() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTask) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  std::string label_;
  AcceleratorService::Token accToken_;

  edm::EDGetTokenT<OutputType> srcToken_;
  bool showResult_;

  TestAlgo algo_;
};


TestAcceleratorServiceProducerGPU::TestAcceleratorServiceProducerGPU(const edm::ParameterSet& iConfig):
  label_(iConfig.getParameter<std::string>("@module_label")),
  accToken_(edm::Service<AcceleratorService>()->book()),
  showResult_(iConfig.getUntrackedParameter<bool>("showResult"))
{
  auto srcTag = iConfig.getParameter<edm::InputTag>("src");
  if(!srcTag.label().empty()) {
    srcToken_ = consumes<OutputType>(srcTag);
  }

  produces<OutputType>();
}

void TestAcceleratorServiceProducerGPU::acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  const OutputType *input = nullptr;
  if(!srcToken_.isUninitialized()) {
    edm::Handle<OutputType> hin;
    iEvent.getByToken(srcToken_, hin);
    input = hin.product();
  }

  algo_.setInput(input, iEvent.id().event(), iEvent.streamID());
  
  edm::LogPrint("Foo") << "TestAcceleratorServiceProducerGPU::acquire begin event " << iEvent.id().event() << " stream " << iEvent.streamID() << " label " << label_ << " input " << input;
  edm::Service<AcceleratorService> acc;
  acc->schedule(accToken_, iEvent.streamID(), std::move(waitingTaskHolder), input,
                accelerator::algoGPUCuda(&algo_),
                accelerator::algoCPU(&algo_)
                );

  edm::LogPrint("Foo") << "TestAcceleratorServiceProducerGPU::acquire end event " << iEvent.id().event() << " stream " << iEvent.streamID() << " label " << label_;
}

void TestAcceleratorServiceProducerGPU::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogPrint("Foo") << "TestAcceleratorServiceProducerGPU::produce begin event " << iEvent.id().event() << " stream " << iEvent.streamID() << " label " << label_;
  std::unique_ptr<OutputType> ret;
  if(algo_.ranOnGPU()) {
    ret = std::make_unique<OutputType>(algo_.getGPUOutput(), algo_.makeTransfer());
  }
  else {
    ret = std::make_unique<OutputType>(algo_.getOutput());
  }

  unsigned int value = showResult_ ? ret->getCPUProduct() : 0;
  edm::LogPrint("Foo") << "TestAcceleratorServiceProducerGPU::produce end event " << iEvent.id().event() << " stream " << iEvent.streamID() << " label " << label_ << " result " << value;
  iEvent.put(std::move(ret));
}

void TestAcceleratorServiceProducerGPU::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag());
  desc.addUntracked<bool>("showResult", false);
  descriptions.add("testAcceleratorServiceProducerGPU", desc);
}

DEFINE_FWK_MODULE(TestAcceleratorServiceProducerGPU);
