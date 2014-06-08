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
    ThreadWork(const edm::ParameterSet& iConfig):
      m_id(iConfig.getParameter<int>("id"))
    {}

    int id() const { return m_id; }

    std::mutex& mutex() { return m_mutex; }
    std::condition_variable& cv() { return m_cv; }

    void run() {
      {
        edm::LogPrint("Test") << "Id " << m_id << " thread " << std::this_thread::get_id();
        std::lock_guard<std::mutex> lk(m_mutex);
        edm::LogPrint("Test") << "Id " << m_id << " thread " << std::this_thread::get_id() << " got lock";
        double sum = doWork();
        edm::LogPrint("Test") << "Id " << m_id << " thread " << std::this_thread::get_id() << " sum " << sum;
      }
      m_cv.notify_one();
      edm::LogPrint("Test") << "Id " << m_id << " thread " << std::this_thread::get_id() << " cv signaled";
    }
  private:
    std::mutex m_mutex;
    std::condition_variable m_cv;
    const int m_id;
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

    const ThreadWork& threadWork() const { return *m_threadWork; }
    
    mutable std::atomic<unsigned int> count;
  private:
    std::shared_ptr<ThreadWork> m_threadWork;
    std::thread m_masterThread;
  };

  class OscarMTTest: public edm::stream::EDProducer<
    edm::GlobalCache<MasterThread>
    > {
  public:
    explicit OscarMTTest(const edm::ParameterSet& iConfig, const MasterThread *masterThread) {}
    virtual ~OscarMTTest() {}

    static std::unique_ptr<MasterThread> initializeGlobalCache(const edm::ParameterSet& iConfig) {
      auto work = std::make_shared<ThreadWork>(iConfig);

      std::unique_lock<std::mutex> lk(work->mutex());
      edm::LogPrint("Test") << "Id " << work->id() << " initializeGlobalCache(): created lock";

      auto masterThread = std::unique_ptr<MasterThread>(new MasterThread(work));
      edm::LogPrint("Test") << "Id " << work->id() << " initializeGlobalCache(): Created MasterThread, waiting on cv";
      work->cv().wait(lk);
      edm::LogPrint("Test") << "Id " << work->id() << " initializeGlobalCache(): Wait finished";
      lk.unlock();

      return masterThread;
    }

    static void globalEndJob(MasterThread *master) {
      edm::LogPrint("Test") << "Id " << master->threadWork().id() << " number of events seen " << master->count;
    }

    virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  private:
  };


  void OscarMTTest::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    ++(globalCache()->count);

    double sum = doWork();

    edm::LogPrint("Test") << "Id " << globalCache()->threadWork().id()
                          << " event " << iEvent.id().event() 
                          << " stream " << iEvent.streamID().value()
                          << " " << sum;
  }
}

DEFINE_FWK_MODULE(OscarMTTest);
