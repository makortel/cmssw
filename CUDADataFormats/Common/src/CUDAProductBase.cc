#include "CUDADataFormats/Common/interface/CUDAProductBase.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

CUDAProductBase::~CUDAProductBase() {
  // TODO: a callback notifying a WaitingTaskHolder would avoid
  // blocking the CPU, but would also require more work.
  if(stream_) {
    stream_->synchronize();
  }
}

bool CUDAProductBase::isAvailable() const {
  // In absence of event, the product was available already at the end
  // of produce() of the producer.
  if(not event_) {
    return true;
  }
  return event_->has_occurred();
}
