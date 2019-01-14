#ifndef HeterogeneousCore_CUDAServices_CUDAService_h
#define HeterogeneousCore_CUDAServices_CUDAService_h

#include <utility>
#include <vector>

#include <cuda/api_wrappers.h>

#include "FWCore/Utilities/interface/StreamID.h"

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

namespace edm {
  class ParameterSet;
  class ActivityRegistry;
  class ConfigurationDescriptions;
}

/**
 * TODO:
 * - CUDA stream management?
 *   * Not really needed until we want to pass CUDA stream objects from one module to another
 *   * Which is not really needed until we want to go for "streaming mode"
 *   * Until that framework's inter-module synchronization is safe (but not necessarily optimal)
 * - Management of (preallocated) memory?
 */
class CUDAService {
public:
  CUDAService(edm::ParameterSet const& iConfig, edm::ActivityRegistry& iRegistry);
  ~CUDAService();

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

  // To be used in global context when an edm::Stream is not available
  bool enabled() const { return enabled_; }
  // To be used in stream context when an edm::Stream is available
  bool enabled(edm::StreamID streamId) const { return enabled(static_cast<unsigned int>(streamId)); }
  bool enabled(unsigned int streamId) const { return enabled_ && (numberOfStreamsTotal_ == 0 || streamId < numberOfStreamsTotal_); } // to make testing easier

  int numberOfDevices() const { return numberOfDevices_; }

  // major, minor
  std::pair<int, int> computeCapability(int device) { return computeCapabilities_.at(device); }

  // Returns the id of device with most free memory. If none is found, returns -1.
  int deviceWithMostFreeMemory() const;

  // Set the current device
  void setCurrentDevice(int device) const;

  // Get the current device
  int getCurrentDevice() const;

  // Allocate device memory
  template <typename T>
  typename cudautils::device::impl::make_device_unique_selector<T>::non_array
  make_device_unique(cuda::stream_t<>& stream) {
    static_assert(std::is_trivially_constructible<T>::value, "Allocating with non-trivial constructor on the device memory is not supported");
    int dev = getCurrentDevice();
    void *mem = allocate_device(dev, sizeof(T), stream);
    return typename cudautils::device::impl::make_device_unique_selector<T>::non_array(reinterpret_cast<T *>(mem),
                                                                                       cudautils::device::impl::DeviceDeleter([this, dev](void *ptr) {
                                                                                           this->free_device(dev, ptr);
                                                                                         }));
  }

  template <typename T>
  typename cudautils::device::impl::make_device_unique_selector<T>::unbounded_array
  make_device_unique(size_t n, cuda::stream_t<>& stream) {
    using element_type = typename std::remove_extent<T>::type;
    static_assert(std::is_trivially_constructible<element_type>::value, "Allocating with non-trivial constructor on the device memory is not supported");
    int dev = getCurrentDevice();
    void *mem = allocate_device(dev, n*sizeof(element_type), stream);
    return typename cudautils::device::impl::make_device_unique_selector<T>::unbounded_array(reinterpret_cast<element_type *>(mem),
                                                                                             cudautils::device::impl::DeviceDeleter([this, dev](void *ptr) {
                                                                                                 this->free_device(dev, ptr);
                                                                                               }));
  }

  template <typename T, typename ...Args>
  typename cudautils::device::impl::make_device_unique_selector<T>::bounded_array
  make_device_unique(Args&&...) = delete;

  // Allocate pinned host memory
  template <typename T>
  typename cudautils::host::impl::make_host_unique_selector<T>::non_array
  make_host_unique(cuda::stream_t<>& stream) {
    static_assert(std::is_trivially_constructible<T>::value, "Allocating with non-trivial constructor on the pinned host memory is not supported");
    void *mem = allocate_host(sizeof(T), stream);
    return typename cudautils::host::impl::make_host_unique_selector<T>::non_array(reinterpret_cast<T *>(mem),
                                                                                   cudautils::host::impl::HostDeleter([this](void *ptr) {
                                                                                       this->free_host(ptr);
                                                                                     }));
  }

  template <typename T>
  typename cudautils::host::impl::make_host_unique_selector<T>::unbounded_array
  make_host_unique(size_t n, cuda::stream_t<>& stream) {
    using element_type = typename std::remove_extent<T>::type;
    static_assert(std::is_trivially_constructible<element_type>::value, "Allocating with non-trivial constructor on the pinned host memory is not supported");
    void *mem = allocate_host(n*sizeof(element_type), stream);
    return typename cudautils::host::impl::make_host_unique_selector<T>::unbounded_array(reinterpret_cast<element_type *>(mem),
                                                                                         cudautils::host::impl::HostDeleter([this](void *ptr) {
                                                                                             this->free_host(ptr);
                                                                                           }));
  }

  template <typename T, typename ...Args>
  typename cudautils::host::impl::make_host_unique_selector<T>::bounded_array
  make_host_unique(Args&&...) = delete;
  
  // Free device memory (to be called from unique_ptr)
  void free_device(int device, void *ptr);

  // Free pinned host memory (to be called from unique_ptr)
  void free_host(void *ptr);

private:
  // PIMPL to hide details of allocator
  struct Allocator;
  std::unique_ptr<Allocator> allocator_;
  void *allocate_device(int dev, size_t nbytes, cuda::stream_t<>& stream);
  void *allocate_host(size_t nbytes, cuda::stream_t<>& stream);

  int numberOfDevices_ = 0;
  unsigned int numberOfStreamsTotal_ = 0;
  std::vector<std::pair<int, int>> computeCapabilities_;
  bool enabled_ = false;
};

#endif
