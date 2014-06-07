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

  struct CondVar {
    std::mutex mutex;
    std::condition_variable cv;
  };

  void threadWork(CondVar& cv) {
    {
      //std::lock_guard<std::mutex>(cv.mutex);
      edm::LogPrint("Test") << "Hello from thread " << std::this_thread::get_id();
      double sum = doWork();
      edm::LogPrint("Test") << "Hello again from thread " << std::this_thread::get_id() << " sum " << sum;
    }
    //cv.cv.notify_one();
  }

  class ThreadWork {
  public:
    ThreadWork(CondVar& cv): m_cv(cv) {};
    void run() {
      {
        std::lock_guard<std::mutex>(m_cv.mutex);
        edm::LogPrint("Test") << "Hello from thread " << std::this_thread::get_id();
        double sum = doWork();
        edm::LogPrint("Test") << "Hello again from thread " << std::this_thread::get_id() << " sum " << sum;
      }
      m_cv.cv.notify_one();
    }
  private:
    CondVar& m_cv;
  };

  class MasterThread {
  public:
    explicit MasterThread(CondVar& cv):
      count(0),
      m_threadWork(cv),
      m_masterThread(&ThreadWork::run, m_threadWork)
    {}
    ~MasterThread() {
      m_masterThread.join();
    }
    
    mutable std::atomic<unsigned int> count;
  private:
    ThreadWork m_threadWork;
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
      CondVar cv;
      std::unique_lock<std::mutex> lk(cv.mutex);

      auto masterThread = std::make_shared<MasterThread>(cv);
      cv.cv.wait(lk);

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
