#ifndef CUDADataFormats_Common_CUDA_h
#define CUDADataFormats_Common_CUDA_h

#include <memory>

#include <cuda/api_wrappers.h>

#include "CUDADataFormats/Common/interface/CUDABase.h"

namespace edm {
  template <typename T> class Wrapper;
}

/**
 * The purpose of this class is to wrap CUDA data to edm::Event in a
 * way which forces correct use of various utilities.
 *
 * The non-default construction has to be done with CUDAScopedContext
 * (in order to properly register the CUDA event).
 *
 * The default constructor is needed only for the ROOT dictionary generation.
 *
 * The CUDA event is in practice needed only for stream-stream
 * synchronization, but someone with long-enough lifetime has to own
 * it. Here is a somewhat natural place. If overhead is too much, we
 * can e.g. make CUDAService own them (creating them on demand) and
 * use them only where synchronization between streams is needed.
 */
template <typename T>
class CUDA: public CUDABase {
public:
  CUDA() = default; // Needed only for ROOT dictionary generation

  CUDA(const CUDA&) = delete;
  CUDA& operator=(const CUDA&) = delete;
  CUDA(CUDA&&) = default;
  CUDA& operator=(CUDA&&) = default;

private:
  friend class CUDAScopedContext;
  friend class edm::Wrapper<CUDA<T>>;

  explicit CUDA(int device, std::shared_ptr<cuda::stream_t<>> stream, T data):
    CUDABase(device, std::move(stream)),
    data_(std::move(data))
  {}

  T data_; //!
};

#endif
