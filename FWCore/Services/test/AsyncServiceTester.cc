#include "FWCore/Concurrency/interface/Async.h"
#include "FWCore/Concurrency/interface/chain_first.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <condition_variable>
#include <mutex>

namespace edmtest {
  class AsyncServiceTesterService {
  public:
    AsyncServiceTesterService(edm::ParameterSet const& iConfig, edm::ActivityRegistry& iRegistry) : continue_{false} {
      iRegistry.watchPreSourceEarlyTermination([this](edm::TerminationOrigin) { release(); });
      iRegistry.watchPreGlobalEarlyTermination(
          [this](edm::GlobalContext const&, edm::TerminationOrigin) { release(); });
      iRegistry.watchPreStreamEarlyTermination(
          [this](edm::StreamContext const&, edm::TerminationOrigin) { release(); });
    }

    void wait() {
      std::unique_lock lk(mutex_);
      cond_.wait(lk, [this]() { return continue_; });
    }

  private:
    void release() {
      std::unique_lock lk(mutex_);
      continue_ = true;
      cond_.notify_all();
    }

    std::mutex mutex_;
    std::condition_variable cond_;
    CMS_THREAD_GUARD(mutex_) bool continue_;
  };

  class AsyncServiceTester : public edm::stream::EDProducer<edm::ExternalWork> {
  public:
    AsyncServiceTester(edm::ParameterSet const& iConfig) : wait_(iConfig.getUntrackedParameter<bool>("wait")) {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.addUntracked("wait", false)
          ->setComment(
              "If true, use AsyncServiceTesterService to wait the async activity until an early termination signal has "
              "been issued");
      descriptions.addDefault(desc);
    }

    void acquire(edm::Event const& iEvent, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder holder) final {
      if (wait_ and iEvent.id().event() >= 2) {
        edm::Service<AsyncServiceTesterService> testService;
        testService->wait();
      }
      if (status_ != 0) {
        throw cms::Exception("Assert") << "In acquire: status_ was " << status_ << ", expected 0";
      }
      edm::Service<edm::Async> as;
      as->run(
          std::move(holder),
          [this]() {
            if (status_ != 0) {
              throw cms::Exception("Assert") << "In async function: status_ was " << status_ << ", expected 0";
            }
            ++status_;
          },
          []() { return "Calling AsyncServiceTester::acquire()"; });
    }

    void produce(edm::Event&, edm::EventSetup const&) final {
      if (status_ != 1) {
        throw cms::Exception("Assert") << "In analyze: status_ was " << status_ << ", expected 1";
      }
      status_ = 0;
    }

  private:
    std::atomic<int> status_ = 0;
    bool wait_;
  };
}  // namespace edmtest

DEFINE_FWK_MODULE(edmtest::AsyncServiceTester);

DEFINE_FWK_SERVICE(edmtest::AsyncServiceTesterService);
