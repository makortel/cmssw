#include "CUDADataFormats/Common/interface/CUDABase.h"

CUDABase::CUDABase(int device, std::shared_ptr<cuda::stream_t<>> stream):
  stream_(std::move(stream)),
  event_(std::make_unique<cuda::event_t>(cuda::event::create(device,
                                                             cuda::event::sync_by_busy_waiting,   // default; we should try to avoid explicit synchronization, so maybe the value doesn't matter much?
                                                             cuda::event::dont_record_timings))), // it should be a bit faster to ignore timings
  device_(device)
{
  // Record CUDA event to the CUDA stream. The event will become
  // "occurred" after all work queued to the stream before this
  // point has been finished.
  event_->record(stream_->id());
}


