#ifndef HeterogeneousCore_Product_interface_HeterogeneousData_h
#define HeterogeneousCore_Product_interface_HeterogeneousData_h

#include "FWCore/Utilities/interface/Exception.h"

#include <bitset>
#include <mutex>

enum class HeterogeneousDevice {
  kCPU = 0,
  kGPUMock,
  kGPUCuda,
  kSize
};

class HeterogeneousDeviceId {
public:
  HeterogeneousDeviceId():
    deviceType_(HeterogeneousDevice::kCPU),
    deviceId_(0)
  {}
  explicit HeterogeneousDeviceId(HeterogeneousDevice device, unsigned int id=0):
    deviceType_(device), deviceId_(id)
  {}

  HeterogeneousDevice deviceType() const { return deviceType_; }

  unsigned int deviceId() const { return deviceId_; }
private:
  HeterogeneousDevice deviceType_;
  unsigned int deviceId_;
};

template <typename CPUProduct, typename GPUProduct>
class HeterogeneousProduct {
public:
  using TransferCallback = std::function<void(const GPUProduct&, CPUProduct&)>;

  HeterogeneousProduct() = default;
  HeterogeneousProduct(CPUProduct&& data):
    cpuProduct_(std::move(data)) {
    location_.set(static_cast<unsigned int>(HeterogeneousDevice::kCPU));
  }
  HeterogeneousProduct(GPUProduct&& data, TransferCallback transfer):
    gpuProduct_(std::move(data)),
    transfer_(std::move(transfer)) {
    location_.set(static_cast<unsigned int>(HeterogeneousDevice::kGPUCuda));
  }


  void swap(HeterogeneousProduct& other) {
    if(this == &other)
      return;

    std::lock(mutex_, other.mutex_);
    std::lock_guard<std::mutex> lk1(mutex_, std::adopt_lock);
    std::lock_guard<std::mutex> lk2(other.mutex_, std::adopt_lock);

    std::swap(cpuProduct_, other.cpuProduct_);
    std::swap(gpuProduct_, other.gpuProduct_);
    std::swap(location_, other.location_);
    std::swap(transfer_, other.transfer_);
  }

  bool isProductOn(HeterogeneousDevice loc) const {
    return location_[static_cast<unsigned int>(loc)];
  }

  const CPUProduct& getCPUProduct() const {
    if(!isProductOn(HeterogeneousDevice::kCPU)) transferToCPU();
    return cpuProduct_;
  }

  const GPUProduct& getGPUProduct() const {
    if(!isProductOn(HeterogeneousDevice::kGPUCuda))
      throw cms::Exception("LogicError") << "Called getGPUProduct(), but the data is not on GPU! Location bitfield is " << location_.to_string();
    return gpuProduct_;
  }

private:
  void transferToCPU() const {
    if(!isProductOn(HeterogeneousDevice::kGPUCuda)) {
      throw cms::Exception("LogicError") << "Called transferToCPU, but the data is not on GPU! Location bitfield is " << location_.to_string();
    }
    
    std::lock_guard<std::mutex> lk(mutex_);
    transfer_(gpuProduct_, cpuProduct_);
    location_.set(static_cast<unsigned int>(HeterogeneousDevice::kCPU));
  }
  
  mutable std::mutex mutex_;
  mutable CPUProduct cpuProduct_;
  GPUProduct gpuProduct_;
  mutable std::bitset<static_cast<unsigned int>(HeterogeneousDevice::kSize)> location_;
  TransferCallback transfer_;
};


#endif
