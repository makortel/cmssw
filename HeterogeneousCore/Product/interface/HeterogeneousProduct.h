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

    const T& product() const { return data_; }
    T& product() { return data_; }
  private:
    T data_;
  };
  template <typename T> struct ProductToEnum<CPUProduct<T>> { static constexpr const HeterogeneousDevice value = HeterogeneousDevice::kCPU; };
  template <typename T> auto cpuProduct(T&& data) { return CPUProduct<T>(std::move(data)); }

  template <typename T>
  class GPUMockProduct {
  public:
    using DataType = T;
    static constexpr const HeterogeneousDevice tag = HeterogeneousDevice::kGPUMock;
    
    GPUMockProduct() = default;
    GPUMockProduct(T&& data): data_(std::move(data)) {}

    void swap(GPUMockProduct<T>& other) {
      if(this == &other) return;
      std::swap(data_, other.data_);
    }

    const T& product() const { return data_; }
    T& product() { return data_; }
private:
    T data_;
  };
  template <typename T> struct ProductToEnum<GPUMockProduct<T>> { static constexpr const HeterogeneousDevice value = HeterogeneousDevice::kGPUMock; };
  template <typename T> auto gpuMockProduct(T&& data) { return GPUMockProduct<T>(std::move(data)); }

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

    const T& product() const { return data_; }
    T& product() { return data_; }
private:
    T data_;
  };
  template <typename T> struct ProductToEnum<GPUCudaProduct<T>> { static constexpr const HeterogeneousDevice value = HeterogeneousDevice::kGPUCuda; };
  template <typename T> auto gpuCudaProduct(T&& data) { return GPUCudaProduct<T>(std::move(data)); }

  // used below for "parameter pack expansion for function argument" trick
  template <typename ...Args> void call_nop(Args&&... args) {}

  // Empty struct for tuple defitionons
  struct Empty {};

  // Metaprogram to return the *Product<T> type for a given enumerator if it exists in Types... pack
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
  using IfInPack_t = typename IfInPack<device, Types...>::type;

  // Metaprogram to construct the callback function type for device->CPU transfers
  template <typename CPUProduct, typename DeviceProduct>
  struct CallBackType {
    using type = std::function<void(typename DeviceProduct::DataType const&, typename CPUProduct::DataType&)>;
  };
  template <typename CPUProduct>
  struct CallBackType<CPUProduct, Empty> {
    using type = Empty;
  };
  template <typename CPUProduct, typename DeviceProductOrEmpty>
  using CallBackType_t = typename CallBackType<CPUProduct, DeviceProductOrEmpty>::type;

  // Metaprogram to loop over two tuples and a bitset (of equal
  // length), and if bitset is set to true call a function from one of
  // the tuples with arguments from the second tuple
  template <typename FunctionTuple, typename ProductTuple, typename BitSet, typename FunctionTupleElement, size_t sizeMinusIndex>
  struct CallFunctionIf {
    static bool call(const FunctionTuple& functionTuple, ProductTuple& productTuple, const BitSet& bitSet) {
      constexpr const auto index = bitSet.size()-sizeMinusIndex;
      if(bitSet[index]) {
        std::get<index>(functionTuple)(std::get<index>(productTuple).product(), std::get<0>(productTuple).product());
        return true;
      }
      return CallFunctionIf<FunctionTuple, ProductTuple, BitSet,
                            std::tuple_element_t<index+1, FunctionTuple>, sizeMinusIndex-1>::call(functionTuple, productTuple, bitSet);
    }
  };
  template <typename FunctionTuple, typename ProductTuple, typename BitSet, size_t sizeMinusIndex>
  struct CallFunctionIf<FunctionTuple, ProductTuple, BitSet, Empty, sizeMinusIndex> {
    static bool call(const FunctionTuple& functionTuple, ProductTuple& productTuple, const BitSet& bitSet) {
      constexpr const auto index = bitSet.size()-sizeMinusIndex;
      return CallFunctionIf<FunctionTuple, ProductTuple, BitSet,
                            std::tuple_element_t<index+1, FunctionTuple>, sizeMinusIndex-1>::call(functionTuple, productTuple, bitSet);
    }
  };
  template <typename FunctionTuple, typename ProductTuple, typename BitSet>
  struct CallFunctionIf<FunctionTuple, ProductTuple, BitSet, Empty, 0> {
    static bool call(const FunctionTuple& functionTuple, ProductTuple& productTuple, const BitSet& bitSet) {
      return false;
    }
  };

  // Metaprogram to specialize getProduct() for CPU
  template <HeterogeneousDevice device>
  struct GetOrTransferProduct {
    template <typename FunctionTuple, typename ProductTuple, typename BitSet>
    static const auto& getProduct(const FunctionTuple& functionTuple, ProductTuple& productTuple, const BitSet& bitSet) {
      constexpr const auto index = static_cast<unsigned int>(device);
      if(!bitSet[index]) {
        throw cms::Exception("LogicError") << "Called getProduct() for device " << index << " but the data is not there! Location bitfield is " << bitSet.to_string();
      }
      return std::get<index>(productTuple).product();
    }
  };

  template <>
  struct GetOrTransferProduct<HeterogeneousDevice::kCPU> {
    template <typename FunctionTuple, typename ProductTuple, typename BitSet>
    static const auto& getProduct(const FunctionTuple& functionTuple, ProductTuple& productTuple, BitSet& bitSet) {
      constexpr const auto index = static_cast<unsigned int>(HeterogeneousDevice::kCPU);
      if(!bitSet[index]) {
        auto found = CallFunctionIf<FunctionTuple, ProductTuple, BitSet,
                                    std::tuple_element_t<1, FunctionTuple>, bitSet.size()-1>::call(functionTuple, productTuple, bitSet);
        if(!found) {
          throw cms::Exception("LogicError") << "Attempted to transfer data to CPU, but the data is not available anywhere! Location bitfield is " << bitSet.to_string();
        }
      }
      bitSet.set(index);
      return std::get<index>(productTuple).product();
    }
  };
}

/**
 * TODO:
 * * extend transfers to device->device (within a single device type)
 */
template <typename CPUProduct, typename... Types>
class HeterogeneousProduct {
  using ProductTuple = std::tuple<CPUProduct,
                                  heterogeneous::IfInPack_t<HeterogeneousDevice::kGPUMock, Types...>,
                                  heterogeneous::IfInPack_t<HeterogeneousDevice::kGPUCuda, Types...>
                                  >;
  using TransferToCPUTuple = std::tuple<heterogeneous::Empty, // no need to transfer from CPU to CPU
                                        heterogeneous::CallBackType_t<CPUProduct, std::tuple_element_t<static_cast<unsigned int>(HeterogeneousDevice::kGPUMock), ProductTuple>>,
                                        heterogeneous::CallBackType_t<CPUProduct, std::tuple_element_t<static_cast<unsigned int>(HeterogeneousDevice::kGPUCuda), ProductTuple>>
                                        >;
  using BitSet = std::bitset<static_cast<unsigned int>(HeterogeneousDevice::kSize)>;

  // Some sanity checks
  static_assert(std::tuple_size<ProductTuple>::value == std::tuple_size<TransferToCPUTuple>::value, "Size mismatch");
  static_assert(std::tuple_size<ProductTuple>::value == static_cast<unsigned int>(HeterogeneousDevice::kSize), "Size mismatch");
public:
  HeterogeneousProduct() = default;
  HeterogeneousProduct(CPUProduct&& data) {
    constexpr const auto index = static_cast<unsigned int>(HeterogeneousDevice::kCPU);
    std::get<index>(products_) = std::move(data);
    location_.set(index);
  }

  template <typename H, typename F>
  HeterogeneousProduct(H&& data, F transferToCPU) {
    constexpr const auto index = static_cast<unsigned int>(heterogeneous::ProductToEnum<std::remove_reference_t<H> >::value);
    static_assert(!std::is_same<std::tuple_element_t<index, ProductTuple>,
                                heterogeneous::Empty>::value,
                  "This HeterogeneousProduct does not support this type");
    std::get<index>(products_) = std::move(data);
    std::get<index>(transfersToCPU_) = std::move(transferToCPU);
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

    std::lock_guard<std::mutex> lk(mutex_);
    return heterogeneous::GetOrTransferProduct<device>::getProduct(transfersToCPU_, products_, location_);
    /*
    if(!isProductOn(device)) {
      throw cms::Exception("LogicError") << "Called getProduct() for device " << index << " but the data is not there! Location bitfield is " << location_.to_string();
    }
    return std::get<index>(products_).product();
    */
  }

private:
  template <std::size_t ...Is>
  void swapTuple(std::index_sequence<Is...>, std::tuple<Types...>& other) {
    call_nop(std::get<Is>(products_).swap(std::get<Is>(other))...);
  }

  /*
  void transferToCPU() const {
    std::lock_guard<std::mutex> lk(mutex_);

    auto found = heterogeneous::CallFunctionIf<ProductTuple, TransferToCPUTuple, BitSet, location_.size()-1>(products_, transfersToCPU_, location_);
    if(!found) {
      throw cms::Exception("LogicError") << "Attempted to transfer data to CPU, but the data is not available anywhere! Location bitfield is " << location_.to_string();
    }
    location_.set(static_cast<unsigned int>(HeterogeneousDevice::kCPU));
  }
  */

  mutable std::mutex mutex_;
  mutable ProductTuple products_;
  TransferToCPUTuple transfersToCPU_;
  mutable BitSet location_;
};

/*
template <typename CPUProduct, typename... Types>
template <>
const auto& HeterogeneousProduct<CPUProduct, Types...>::getProduct<HeterogeneousDevice::kCPU>() const {
  if(!isProductOn(HeterogeneousDevice::kCPU)) {
    transferToCPU();
  }
  return std::get<0>(products_).product();
}
*/

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
