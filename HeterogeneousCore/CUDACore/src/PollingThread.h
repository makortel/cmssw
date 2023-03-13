#ifndef HeterogeneousCore_CUDACore_EventPollingThread_h
#define HeterogeneousCore_CUDACore_EventPollingThread_h

#include "HeterogeneousCore/CUDAUtilities/interface/SharedEventPtr.h"
#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"

#include <tbb/concurrent_queue.h>

#include <atomic>
#include <thread>

namespace cms::cuda {
  namespace impl {
    class PollingThread {
    public:
      PollingThread();
      ~PollingThread();

      void then(SharedEventPtr event, edm::WaitingTaskWithArenaHolder holder);

      void stopThread() {
        stopThread_ = true;
      }

    private:
      void threadLoop();

      std::thread thread_;
      using EventHolder = std::pair<SharedEventPtr, edm::WaitingTaskWithArenaHolder>;
      tbb::concurrent_queue<EventHolder> eventHolders_;
      std::atomic<bool> stopThread_;
    };

    inline PollingThread& getPollingThread() {
      static PollingThread thread;
      return thread;
    }
  }

  inline void eventContinuation(SharedEventPtr event, edm::WaitingTaskWithArenaHolder holder) {
    impl::getPollingThread().then(std::move(event), std::move(holder));
  }
}
#endif
