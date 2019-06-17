#ifndef FWCore_Concurrency_make_waiting_task_with_holder_h
#define FWCore_Concurrency_make_waiting_task_with_holder_h

#include "FWCore/Concurrency/interface/WaitingTask.h"
#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"

namespace edm {
  template <typename ALLOC, typename F>
  auto make_waiting_task_with_holder(ALLOC&& iAlloc, edm::WaitingTaskWithArenaHolder h, F&& f) {
    return make_waiting_task(std::forward<ALLOC>(iAlloc), [holder=std::move(h), func=std::forward<F>(f)](std::exception_ptr const *excptr) mutable {
        if(excptr) {
          holder.doneWaiting(*excptr);
          return;
        }

        try {
          func(holder);
        } catch(...) {
          holder.doneWaiting(std::current_exception());
        }
      });
  }
}

#endif
