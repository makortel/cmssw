#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "PollingThread.h"

//#define CUDADEV_POLLINGTHREAD_THEN

#ifndef CUDADEV_POLLINGTHREAD_SLEEP
#define CUDADEV_POLLINGTHREAD_SLEEP 1us
#endif

namespace cms::cuda::impl {
  PollingThread::PollingThread() : stopThread_{false} { thread_ = std::thread(&PollingThread::threadLoop, this); }

  PollingThread::~PollingThread() {
    if (not stopThread_) {
      stopThread();
    }
    thread_.join();
  }

  void PollingThread::then(SharedEventPtr event, edm::WaitingTaskWithArenaHolder holder) {
#ifdef CUDADEV_POLLINGTHREAD_THEN
    {
      std::vector<EventHolder> incomplete;
      incomplete.reserve(eventHolders_.unsafe_size());
      EventHolder elem;
      while(eventHolders_.try_pop(elem)) {
        try {
          auto eventStatus = cudaEventQuery(elem.first.get());
          if (eventStatus == cudaErrorNotReady) {
            incomplete.emplace_back(std::move(elem));
            continue;
          }
          cudaCheck(eventStatus);
          auto h = elem.second.makeWaitingTaskHolderAndRelease();
          h.doneWaiting(nullptr);
          break;
        } catch(...) {
          if (elem.second.hasTask()) {
            auto h = elem.second.makeWaitingTaskHolderAndRelease();
            h.doneWaiting(std::current_exception());
          }
        }
        break;
      }
      for (auto& elem: incomplete) {
        eventHolders_.push(std::move(elem));
      }
    }
#endif
    eventHolders_.emplace(std::move(event), std::move(holder));
  }

  void PollingThread::threadLoop() {
    while(not stopThread_) {
      {
        std::vector<EventHolder> incomplete;
        incomplete.reserve(eventHolders_.unsafe_size());
        EventHolder elem;
        while(eventHolders_.try_pop(elem)) {
          try {
            auto eventStatus = cudaEventQuery(elem.first.get());
            if (eventStatus == cudaErrorNotReady) {
              incomplete.emplace_back(std::move(elem));
              continue;
            }
            cudaCheck(eventStatus);
            elem.second.doneWaiting(nullptr);
          } catch(...) {
            elem.second.doneWaiting(std::current_exception());
          }
        }
        for (auto& elem: incomplete) {
          eventHolders_.push(std::move(elem));
        }
      }
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(CUDADEV_POLLINGTHREAD_SLEEP);
    }
  }
}
