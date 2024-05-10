#ifndef FWCore_Concurrency_Async_h
#define FWCore_Concurrency_Async_h

#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Concurrency/interface/WaitingThreadPool.h"

namespace edm {
  // All member functions are thread safe
  class Async {
  public:
    Async() = default;
    virtual ~Async();

    template <typename F, typename G>
    void run(WaitingTaskWithArenaHolder holder, F&& func, G&& errorContextFunc) {
      ensureAllowed();
      pool_.runAsync(std::move(holder), std::forward<F>(func), std::forward<G>(errorContextFunc));
    }

  protected:
    virtual void ensureAllowed() const = 0;

  private:
    WaitingThreadPool pool_;
  };
}  // namespace edm

#endif
