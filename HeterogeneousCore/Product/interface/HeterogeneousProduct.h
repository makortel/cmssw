#ifndef HeterogeneousCore_Product_interface_HeterogeneousData_h
#define HeterogeneousCore_Product_interface_HeterogeneousData_h

#include "FWCore/Utilities/interface/Exception.h"

#include <bitset>
#include <mutex>
#include <tuple>

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

namespace heterogeneous {
  template <typename T> struct ProductToEnum {};

  template <typename T>
  class CPUProduct {
  public:
    using DataType = T;
    static constexpr const HeterogeneousDevice tag = HeterogeneousDevice::kCPU;
    
    CPUProduct() = default;
    CPUProduct(T&& data): data_(std::move(data)) {}

    void swap(CPUProduct<T>& other) {
      if(this == &other) return;
      std::swap(data_, other.data_);
    }

    const T& product() const {
      return data_;
    }
  private:
    T data_;
  };
  template <typename T> struct ProductToEnum<CPUProduct<T>> { static constexpr const HeterogeneousDevice value = HeterogeneousDevice::kCPU; };
  template <typename T> auto cpuProduct(T&& data) { return CPUProduct<T>(std::move(data)); }

  template <typename T, typename CPUProduct>
  class GPUMockProduct {
  public:
    using DataType = T;
    static constexpr const HeterogeneousDevice tag = HeterogeneousDevice::kGPUMock;
    using TransferToCPU = std::function<void(const T&, typename CPUProduct::DataType)>;
    
    GPUMockProduct() = default;
    GPUMockProduct(T&& data, TransferToCPU transfer):
      data_(std::move(data)),
      transferToCPU_(std::move(transfer))
    {}

    void swap(GPUMockProduct<T, CPUProduct>& other) {
      if(this == &other) return;
      std::swap(data_, other.data_);
      std::swap(transferToCPU_, other.transferToCPU_);
    }

    const T& product() const {
      return data_;
    }
private:
    T data_;
    TransferToCPU transferToCPU_;
  };
  template <typename T, typename CPUProduct> struct ProductToEnum<GPUMockProduct<T, CPUProduct>> { static constexpr const HeterogeneousDevice value = HeterogeneousDevice::kGPUMock; };
  template <typename T, typename CPUProduct>
  auto gpuMockProduct(T&& data, typename GPUMockProduct<T, CPUProduct>::TransferToCPU transfer) {
    return GPUMockProduct<T, CPUProduct>(std::move(data), std::move(transfer));
  }
    

  template <typename T>
  class GPUCudaProduct {
  public:
    using DataType = T;
    static constexpr const HeterogeneousDevice tag = HeterogeneousDevice::kGPUCuda;
    
    GPUCudaProduct() = default;
    GPUCudaProduct(T&& data): data_(std::move(data)) {}

    void swap(GPUCudaProduct<T>& other) {
      if(this == &other) return;
      std::swap(data_, other.data_);
    }

    const T& product() const {
      return data_;
    }
private:
    T data_;
  };
  template <typename T> struct ProductToEnum<GPUCudaProduct<T>> { static constexpr const HeterogeneousDevice value = HeterogeneousDevice::kGPUCuda; };
  template <typename T> auto gpuCudaProduct(T&& data) { return GPUCudaProduct<T>(std::move(data)); }

  template <typename ...Args> void call_nop(Args&&... args) {}

  struct Empty {};
 
  template <HeterogeneousDevice device, typename... Types>
  struct IfInPack;

  template <HeterogeneousDevice device, typename Type, typename... Types>
  struct IfInPack<device, Type, Types...> {
    using type = std::conditional_t<device==Type::tag,
                                    Type,
                                    typename IfInPack<device, Types...>::type >;
  };
  template <HeterogeneousDevice device>
  struct IfInPack<device> {
    using type = Empty;
  };

  template <HeterogeneousDevice device, typename... Types>
  using ifInPack_t = typename IfInPack<device, Types...>::type;
}

template <typename CPUProduct, typename... Types>
class HeterogeneousProduct {
  using ProductTuple = std::tuple<CPUProduct,
                                  heterogeneous::ifInPack_t<HeterogeneousDevice::kGPUMock, Types...>,
                                  heterogeneous::ifInPack_t<HeterogeneousDevice::kGPUCuda, Types...>
                                  >;
public:
  HeterogeneousProduct() = default;
  HeterogeneousProduct(CPUProduct&& data) {
    constexpr const auto index = static_cast<unsigned int>(HeterogeneousDevice::kCPU);
    std::get<index>(products_) = std::move(data);
    location_.set(index);
  }

  template <typename H>
  HeterogeneousProduct(H&& data) {
    constexpr const auto index = static_cast<unsigned int>(heterogeneous::ProductToEnum<std::remove_reference_t<H> >::value);
    static_assert(!std::is_same<std::tuple_element_t<index, ProductTuple>,
                                heterogeneous::Empty>::value,
                  "This HeterogeneousProduct does not support this type");
    std::get<index>(products_) = std::move(data);
    location_.set(index);
  }

  void swap(HeterogeneousProduct<CPUProduct, Types...>& other) {
    if(this == &other)
      return;

    std::lock(mutex_, other.mutex_);
    std::lock_guard<std::mutex> lk1(mutex_, std::adopt_lock);
    std::lock_guard<std::mutex> lk2(other.mutex_, std::adopt_lock);

    swapTuple(std::index_sequence_for<Types...>{}, other.products_);
    std::swap(location_, other.location_);
  }

  bool isProductOn(HeterogeneousDevice loc) const {
    return location_[static_cast<unsigned int>(loc)];
  }

  template <HeterogeneousDevice device>
  const auto& getProduct() const {
    constexpr const auto index = static_cast<unsigned int>(device);
    static_assert(!std::is_same<std::tuple_element_t<index, ProductTuple>,
                                heterogeneous::Empty>::value,
                  "This HeterogeneousProduct does not support this type");
    if(!isProductOn(device)) {
      throw cms::Exception("LogicError") << "Called getProduct() for device " << index << " but the data is not there! Location bitfield is " << location_.to_string();
    }
    return std::get<index>(products_).product();
  }

private:
  template <std::size_t ...Is>
  void swapTuple(std::index_sequence<Is...>, std::tuple<Types...>& other) {
    call_nop(std::get<Is>(products_).swap(std::get<Is>(other))...);
  }
  
  mutable std::mutex mutex_;
  mutable ProductTuple products_;
  mutable std::bitset<static_cast<unsigned int>(HeterogeneousDevice::kSize)> location_;
};

/*
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
*/

#endif
