#ifndef HeterogenousCore_CUDATest_SimOperationsService_h
#define HeterogenousCore_CUDATest_SimOperationsService_h

#include "HeterogeneousCore/CUDAUtilities/interface/host_noncached_unique_ptr.h"

#include <unordered_map>
#include <vector>

#include <boost/property_tree/ptree.hpp>

#include <cuda/api_wrappers.h>

namespace edm {
  class ParameterSet;
  class ActivityRegistry;
  class ConfigurationDescriptions;
}  // namespace edm

namespace cudatest {
  class TimeCruncher;
  class GPUTimeCruncher;

  class OperationState;
  class OperationBase;
  class OperationCPU;
}


class SimOperationsService {
  using OpVectorCPU = std::vector<std::unique_ptr<cudatest::OperationCPU>>;
  using OpVectorGPU = std::vector<std::unique_ptr<cudatest::OperationBase>>;
public:
  class GPUData {
  public:
    GPUData() = default;
    explicit GPUData(const OpVectorGPU& ops, const unsigned int gangSize);
    ~GPUData();

    GPUData(const GPUData&) = delete;
    GPUData& operator=(const GPUData&) = delete;
    GPUData(GPUData&&);
    GPUData& operator=(GPUData&&);

    cudatest::OperationState makeState();

  private:
    void swap(GPUData& rhs);

    float* kernel_data_d_ = nullptr;

    // These are indexed by the operation index in ops of Service
    // They may to contain null elements
    std::vector<char* > data_d_src_;
    std::vector<cudautils::host::noncached::unique_ptr<char[]>> data_h_src_;
  };

  class AcquireCPUProcessor {
  public:
    AcquireCPUProcessor() = default;

    size_t events() const { return events_; }
    void process(const std::vector<size_t>& indices) const {
      if(index_ >= 0) {
        sos_->acquireCPU(index_, indices);
      }
    }
  private:
    friend class SimOperationsService;
    AcquireCPUProcessor(int i, const SimOperationsService* sos, unsigned int events): sos_{sos}, index_{i}, events_{events} {}
    const SimOperationsService* sos_ = nullptr;
    int index_ = -1;
    unsigned int events_ = 0;
  };

  class AcquireGPUProcessor {
  public:
    AcquireGPUProcessor() = default;

    size_t events() const { return events_; }
    void process(const std::vector<size_t>& indices, cuda::stream_t<>& stream);
  private:
    friend class SimOperationsService;
    AcquireGPUProcessor(int i, const SimOperationsService* sos, GPUData data, unsigned int events): data_{std::move(data)}, sos_{sos}, index_{i}, events_{events} {}
    GPUData data_;
    const SimOperationsService* sos_ = nullptr;
    int index_ = -1;
    unsigned int events_ = 0;
  };

  class ProduceCPUProcessor {
  public:
    ProduceCPUProcessor() = default;

    size_t events() const { return events_; }
    void process(const std::vector<size_t>& indices) const {
      if(index_ >= 0) {
        sos_->produceCPU(index_, indices);
      }
    }

  private:
    friend class SimOperationsService;
    ProduceCPUProcessor(int i, const SimOperationsService* sos, unsigned int events): sos_{sos}, index_{i}, events_{events} {}
    const SimOperationsService* sos_ = nullptr;
    int index_ = -1;
    unsigned int events_ = 0;
  };

  class ProduceGPUProcessor {
  public:
    ProduceGPUProcessor() = default;

    size_t events() const { return events_; }
    void process(const std::vector<size_t>& indices, cuda::stream_t<>& stream);
  private:
    friend class SimOperationsService;
    ProduceGPUProcessor(int i, const SimOperationsService* sos, GPUData data, unsigned int events): data_{std::move(data)}, sos_{sos}, index_{i}, events_{events} {}
    GPUData data_;
    const SimOperationsService* sos_ = nullptr;
    int index_ = -1;
    unsigned int events_ = 0;
  };

  SimOperationsService(edm::ParameterSet const& iConfig, edm::ActivityRegistry& iRegistry);
  ~SimOperationsService();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  unsigned int gangSize() const { return gangSize_; }
  unsigned int numberOfGangs() const { return gangNum_; }
  int maxEvents() { return maxEvents_; }

  AcquireCPUProcessor acquireCPUProcessor(const std::string& moduleLabel) const;
  AcquireGPUProcessor acquireGPUProcessor(const std::string& moduleLabel) const;
  AcquireGPUProcessor acquireGPUProcessor(const std::string& moduleLabel, int gangSize) const; // override the gang size, if you know what you're doing
  ProduceCPUProcessor produceCPUProcessor(const std::string& moduleLabel) const;
  ProduceGPUProcessor produceGPUProcessor(const std::string& moduleLabel) const;

private:
  void acquireCPU(int modIndex, const std::vector<size_t>& indices) const;
  void acquireGPU(int modIndex, const std::vector<size_t>& indices, cudatest::OperationState& state, cuda::stream_t<>& stream) const;
  void produceCPU(int modIndex, const std::vector<size_t>& indices) const;
  void produceGPU(int modIndex, const std::vector<size_t>& indices, cudatest::OperationState& state, cuda::stream_t<>& stream) const;

  const unsigned int gangSize_;
  const unsigned int gangNum_;
  const int maxEvents_;

  std::unique_ptr<cudatest::TimeCruncher> cpuCruncher_;
  std::unique_ptr<cudatest::GPUTimeCruncher> gpuCruncher_;

  std::vector<std::pair<std::string, OpVectorCPU>> acquireOpsCPU_;
  std::vector<std::pair<std::string, OpVectorGPU>> acquireOpsGPU_;
  std::vector<std::pair<std::string, OpVectorCPU>> produceOpsCPU_;
  std::vector<std::pair<std::string, OpVectorGPU>> produceOpsGPU_;
};

#endif
