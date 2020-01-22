#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimOperationsService.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

class TestCUDAProducerSimEWSleeping : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit TestCUDAProducerSimEWSleeping(const edm::ParameterSet& iConfig);
  ~TestCUDAProducerSimEWSleeping();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  void threadWork();
  void sleepFor(std::chrono::nanoseconds t);

  std::vector<edm::EDGetTokenT<int>> srcTokens_;
  edm::EDPutTokenT<int> dstToken_;

  SimOperationsService::AcquireCPUProcessor acquireOpsCPU_;
  SimOperationsService::ProduceCPUProcessor produceOpsCPU_;

  std::mutex mutex_;
  std::condition_variable condition_;
  bool startProcessing_ = false;
  std::atomic<bool> stopProcessing_ = false;
  std::chrono::nanoseconds sleepingTime_;
  edm::WaitingTaskWithArenaHolder holder_;
  std::unique_ptr<std::thread> sleeperThread_;
};

TestCUDAProducerSimEWSleeping::TestCUDAProducerSimEWSleeping(const edm::ParameterSet& iConfig) {
  const auto moduleLabel = iConfig.getParameter<std::string>("@module_label");
  edm::Service<SimOperationsService> sos;
  acquireOpsCPU_ = sos->acquireCPUProcessor(moduleLabel);
  produceOpsCPU_ = sos->produceCPUProcessor(moduleLabel);

  if (acquireOpsCPU_.events() == 0) {
    throw cms::Exception("Configuration") << "Got 0 events, which makes this module useless";
  }
  const auto nevents = acquireOpsCPU_.events();

  if (nevents != produceOpsCPU_.events() and produceOpsCPU_.events() > 0) {
    throw cms::Exception("Configuration")
        << "Got " << nevents << " events for acquire and " << produceOpsCPU_.events() << " for produce CPU";
  }

  for (const auto& src : iConfig.getParameter<std::vector<edm::InputTag>>("srcs")) {
    srcTokens_.emplace_back(consumes<int>(src));
  }

  if (iConfig.getParameter<bool>("produce")) {
    dstToken_ = produces<int>();
  }

  sleeperThread_ = std::make_unique<std::thread>([this]() { threadWork(); });
}

void TestCUDAProducerSimEWSleeping::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("srcs", std::vector<edm::InputTag>{});
  desc.add<bool>("produce", false);

  //desc.add<bool>("useCachingAllocator", true);
  descriptions.addWithDefaultLabel(desc);
}

TestCUDAProducerSimEWSleeping::~TestCUDAProducerSimEWSleeping() {
  if (sleeperThread_) {
    stopProcessing_ = true;
    condition_.notify_one();
    sleeperThread_->join();
  }
}

void TestCUDAProducerSimEWSleeping::threadWork() {
  while (not stopProcessing_.load()) {
    std::chrono::nanoseconds time;
    edm::WaitingTaskWithArenaHolder holder;
    {
      std::unique_lock<std::mutex> lk{mutex_};
      condition_.wait(lk, [this]() { return startProcessing_ or stopProcessing_.load(); });
      if (stopProcessing_.load()) {
        LogTrace("foo") << "Stopping processing";
        break;
      }
      startProcessing_ = false;
      time = sleepingTime_;
      holder = std::move(holder_);
    }
    LogTrace("foo") << "Sleeping for " << time.count() << " ns";
    std::this_thread::sleep_for(time);
    LogTrace("foo") << "Finished, signalling holder";
    holder.doneWaiting(nullptr);
  }
}

void TestCUDAProducerSimEWSleeping::sleepFor(std::chrono::nanoseconds t) {
  {
    std::lock_guard<std::mutex> lk{mutex_};
    sleepingTime_ = t;
    startProcessing_ = true;
  }
  condition_.notify_one();
}

void TestCUDAProducerSimEWSleeping::acquire(const edm::Event& iEvent,
                                            const edm::EventSetup& iSetup,
                                            edm::WaitingTaskWithArenaHolder h) {
  // to make sure the dependencies are set correctly
  for (const auto& t : srcTokens_) {
    iEvent.get(t);
  }

  if (acquireOpsCPU_.events() > 0) {
    // TODO: there can be at most one sleep operation...
    {
      std::lock_guard<std::mutex> lk{mutex_};
      holder_ = std::move(h);
    }
    acquireOpsCPU_.process(std::vector<size_t>{iEvent.id().event() % acquireOpsCPU_.events()},
                           [this](std::chrono::nanoseconds time) { sleepFor(time); });
  }
}

void TestCUDAProducerSimEWSleeping::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (produceOpsCPU_.events() > 0) {
    produceOpsCPU_.process(std::vector<size_t>{iEvent.id().event() % produceOpsCPU_.events()});
  }

  if (not dstToken_.isUninitialized()) {
    iEvent.emplace(dstToken_, 42);
  }
}

DEFINE_FWK_MODULE(TestCUDAProducerSimEWSleeping);
