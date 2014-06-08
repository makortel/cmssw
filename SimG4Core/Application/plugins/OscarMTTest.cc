#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cmath>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace {
  double doWork() {
    double sum = 0.0;

    for(size_t i=0; i<100000; ++i) {
      for(size_t j=0; j<20000; ++j) {
        sum += double(i)*double(j) - std::sin(i)*double(j);
      }
    }
    return sum;
  }

  class ThreadWork {
  public:
    ThreadWork() {};

    std::mutex& mutex() { return m_mutex; }
    std::condition_variable& cv() { return m_cv; }

    void run() {
      {
        edm::LogPrint("Test") << "Hello from thread " << std::this_thread::get_id();
        std::lock_guard<std::mutex> lk(m_mutex);
        edm::LogPrint("Test") << "Hello from thread " << std::this_thread::get_id() << " got lock";
        double sum = doWork();
        edm::LogPrint("Test") << "Hello from thread " << std::this_thread::get_id() << " sum " << sum;
      }
      m_cv.notify_one();
      edm::LogPrint("Test") << "Hello from thread " << std::this_thread::get_id() << " cv signaled";
    }
  private:
    std::mutex m_mutex;
    std::condition_variable m_cv;
  };

  class MasterThread {
  public:
    explicit MasterThread(std::shared_ptr<ThreadWork> work):
      count(0),
      m_threadWork(work),
      m_masterThread([work](){
          work->run();
        })
    {}
    ~MasterThread() {
      m_masterThread.join();
    }
    
    mutable std::atomic<unsigned int> count;
  private:
    std::shared_ptr<ThreadWork> m_threadWork;
    std::thread m_masterThread;
  };

  class OscarMTTest: public edm::stream::EDProducer<
    edm::RunCache<MasterThread>
    > {
  public:
    explicit OscarMTTest(const edm::ParameterSet& iConfig):
      m_id(iConfig.getParameter<int>("id"))
    {}
    virtual ~OscarMTTest() {}

    static std::shared_ptr<MasterThread> globalBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup, const GlobalCache *iCache) {
      auto work = std::make_shared<ThreadWork>();

      std::unique_lock<std::mutex> lk(work->mutex());
      edm::LogPrint("Test") << "globalBeginRun(): created lock";

      auto masterThread = std::make_shared<MasterThread>(work);
      edm::LogPrint("Test") << "globalBeginRun(): Created MasterThread, waiting on cv";
      work->cv().wait(lk);
      edm::LogPrint("Test") << "globalBeginRun(): Wait finished";
      lk.unlock();

      return masterThread;
    };

    static void globalEndRun(const edm::Run& iRun, const edm::EventSetup& iSetup, const RunContext *iContext) {
      edm::LogPrint("Test") << "Number of events seen " << iContext->run()->count;
    }

    virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  private:
    const int m_id;
  };


  void OscarMTTest::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    ++(runCache()->count);

    double sum = doWork();

    edm::LogPrint("Test") << "Id " << m_id 
                          << " event " << iEvent.id().event() 
                          << " stream " << iEvent.streamID().value()
                          << " " << sum;
  }
}

DEFINE_FWK_MODULE(OscarMTTest);
