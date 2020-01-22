#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_noncached_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "SimOperationsService.h"
#include "TimeCruncher.h"
#include "GPUTimeCruncher.h"

#include <mutex>
#include <random>

#include <boost/property_tree/json_parser.hpp>

namespace cms::cudatest {
  struct OperationState {
    OperationState(float* kernel_data,
                   std::vector<cms::cuda::host::noncached::unique_ptr<char[]>>& h_src,
                   std::vector<char*> d_src)
        : kernel_data_d{kernel_data}, data_d_src{std::move(d_src)} {
      data_h_src.resize(h_src.size());
      std::transform(h_src.begin(), h_src.end(), data_h_src.begin(), [](auto& ptr) { return ptr.get(); });
    }

    float* kernel_data_d;

    std::vector<char*> data_h_src;
    std::vector<char*> data_d_src;

    std::vector<cms::cuda::host::unique_ptr<char[]>> data_h_dst;
    std::vector<cms::cuda::device::unique_ptr<char[]>> data_d_dst;

    int opIndex = 0;
  };

  class OperationBase {
  public:
    virtual ~OperationBase() = default;

    virtual size_t size() const = 0;

    virtual unsigned int maxBytesToDevice() const { return 0; }
    virtual unsigned int maxBytesToHost() const { return 0; }

    virtual void operate(const std::vector<size_t>& indices, OperationState& state, cudaStream_t stream) const {
      throw cms::Exception("NotImplemented") << "OperationBase::operate()";
    }
  };
}  // namespace cms::cudatest

namespace {
  class OperationTime : public cms::cudatest::OperationBase {
  public:
    explicit OperationTime(const boost::property_tree::ptree& conf, const double gangFactor = 1.0)
        : gangFactor_{gangFactor} {
      if (auto unit = conf.get<std::string>("unit"); unit != "ns") {
        throw cms::Exception("Configuration") << "OperationTime expected 'ns', got '" << unit << "'";
      }
      for (auto elem : conf.get_child("values")) {
        time_.emplace_back(elem.second.get<unsigned long long>(""));
      }
    }

    size_t size() const override { return time_.size(); }

  protected:
    std::chrono::nanoseconds totalTime(const std::vector<size_t>& indices) const {
      std::chrono::nanoseconds total{};
      std::chrono::nanoseconds max{};
      for (size_t i : indices) {
        total += time_[i];
        max = std::max(max, time_[i]);
      }
      return max + std::chrono::nanoseconds(static_cast<long int>((total - max).count() * gangFactor_));
    }

  private:
    std::vector<std::chrono::nanoseconds> time_;
    const double gangFactor_;
  };
}  // namespace

namespace cms::cudatest {
  class OperationCPU : public OperationTime {
  public:
    explicit OperationCPU(const boost::property_tree::ptree& conf, const cms::cudatest::TimeCruncher* cruncher)
        : OperationTime(conf), cruncher_(cruncher) {}

    void operate(const std::vector<size_t>& indices) const { cruncher_->crunch_for(totalTime(indices)); }

    virtual void operate(const std::vector<size_t>& indices, const SleepFunction& sleep) const { operate(indices); }

  private:
    const TimeCruncher* cruncher_;
  };
}  // namespace cms::cudatest

namespace {
  class OperationSleep final : public cms::cudatest::OperationCPU {
  public:
    explicit OperationSleep(const boost::property_tree::ptree& conf, const cms::cudatest::TimeCruncher* cruncher)
        : OperationCPU(conf, cruncher) {}

    using OperationCPU::operate;

    void operate(const std::vector<size_t>& indices, const cms::cudatest::SleepFunction& sleep) const override {
      sleep(totalTime(indices));
    }
  };

  class OperationKernel final : public OperationTime {
  public:
    explicit OperationKernel(const boost::property_tree::ptree& conf,
                             const cms::cudatest::GPUTimeCruncher* cruncher,
                             const double gangKernelFactor)
        : OperationTime(conf, gangKernelFactor), gpuCruncher_(cruncher) {}

    void operate(const std::vector<size_t>& indices,
                 cms::cudatest::OperationState& state,
                 cudaStream_t stream) const override {
      gpuCruncher_->crunch_for(totalTime(indices), state.kernel_data_d, stream);
    };

  private:
    const cms::cudatest::GPUTimeCruncher* gpuCruncher_;
  };

  std::mutex g_fakeCudaMutex;
  class OperationFake final : public OperationTime {
  public:
    explicit OperationFake(const boost::property_tree::ptree& conf,
                           const cms::cudatest::TimeCruncher* cruncher,
                           bool useLocks)
        : OperationTime(conf), cruncher_(cruncher), useLocks_(useLocks) {}

    void operate(const std::vector<size_t>& indices,
                 cms::cudatest::OperationState& state,
                 cudaStream_t stream) const override {
      std::unique_lock lock{g_fakeCudaMutex, std::defer_lock};
      if (useLocks_) {
        lock.lock();
      }
      cruncher_->crunch_for(totalTime(indices));
    }

  private:
    const cms::cudatest::TimeCruncher* cruncher_;
    const bool useLocks_;
  };

  class OperationBytes : public cms::cudatest::OperationBase {
  public:
    explicit OperationBytes(const boost::property_tree::ptree& conf) {
      if (auto unit = conf.get<std::string>("unit"); unit != "bytes") {
        throw cms::Exception("Configuration") << "OperationBytes expected 'bytes', got '" << unit << "'";
      }
      for (auto elem : conf.get_child("values")) {
        bytes_.emplace_back(elem.second.get<unsigned int>(""));
      }
    }

    size_t size() const override { return bytes_.size(); }

  protected:
    unsigned long int totalBytes(const std::vector<size_t>& indices) const {
      unsigned long total = 0;
      for (size_t i : indices) {
        total += bytes_[i];
      }
      return total;
    }

    unsigned long int maxBytes() const {
      auto maxel = std::max_element(bytes_.begin(), bytes_.end());
      if (maxel == bytes_.end())
        return 0;
      return *maxel;
    }

  private:
    std::vector<unsigned long int> bytes_;
  };

  class OperationMemcpyToDevice final : public OperationBytes {
  public:
    explicit OperationMemcpyToDevice(const boost::property_tree::ptree& conf) : OperationBytes(conf) {}

    unsigned int maxBytesToDevice() const override { return maxBytes(); }

    void operate(const std::vector<size_t>& indices,
                 cms::cudatest::OperationState& state,
                 cudaStream_t stream) const override {
      const auto i = state.opIndex;

      const auto bytes = totalBytes(indices);

      auto data_d = cms::cuda::make_device_unique<char[]>(bytes, stream);
      LogTrace("foo") << "MemcpyToDevice " << i << " from " << static_cast<const void*>(state.data_h_src[i]) << " to "
                      << static_cast<const void*>(data_d.get()) << " " << bytes << " bytes";
      cudaCheck(cudaMemcpyAsync(data_d.get(), state.data_h_src[i], bytes, cudaMemcpyHostToDevice, stream));
      state.data_d_dst.emplace_back(std::move(data_d));
    }
  };

  class OperationMemcpyToHost final : public OperationBytes {
  public:
    explicit OperationMemcpyToHost(const boost::property_tree::ptree& conf) : OperationBytes(conf) {}

    unsigned int maxBytesToHost() const override { return maxBytes(); }

    void operate(const std::vector<size_t>& indices,
                 cms::cudatest::OperationState& state,
                 cudaStream_t stream) const override {
      const auto i = state.opIndex;

      const auto bytes = totalBytes(indices);

      auto data_h = cms::cuda::make_host_unique<char[]>(bytes, stream);
      LogTrace("foo") << "MemcpyToHost " << i << " from " << static_cast<const void*>(state.data_d_src[i]) << " to "
                      << static_cast<const void*>(data_h.get()) << " " << bytes << " bytes";
      cudaCheck(cudaMemcpyAsync(data_h.get(), state.data_d_src[i], bytes, cudaMemcpyHostToDevice, stream));
      state.data_h_dst.emplace_back(std::move(data_h));
    }
  };

  class OperationMemset final : public OperationBytes {
  public:
    explicit OperationMemset(const boost::property_tree::ptree& conf) : OperationBytes(conf) {}

    unsigned int maxBytesToHost() const override { return maxBytes(); }

    void operate(const std::vector<size_t>& indices,
                 cms::cudatest::OperationState& state,
                 cudaStream_t stream) const override {
      const auto i = state.opIndex;

      const auto bytes = totalBytes(indices);

      LogTrace("foo") << "Memset " << i << " on 0x" << static_cast<const void*>(state.data_d_src[i]) << " " << bytes
                      << " bytes";
      cudaCheck(cudaMemsetAsync(state.data_d_src[i], 42, bytes, stream));
    }
  };
}  // namespace

SimOperationsService::SimOperationsService(edm::ParameterSet const& iConfig, edm::ActivityRegistry& iRegistry)
    : gangSize_{iConfig.getParameter<unsigned int>("gangSize")},
      gangNum_{iConfig.getParameter<unsigned int>("gangNumber")},
      maxEvents_{iConfig.getParameter<int>("maxEvents")} {
  if (gangSize_ == 0) {
    throw cms::Exception("Configuration") << "gangSize must be larger than 0";
  }

  {
    boost::property_tree::ptree calib_root_node;
    boost::property_tree::read_json(iConfig.getParameter<edm::FileInPath>("cpuCalibration").fullPath(),
                                    calib_root_node);
    std::vector<unsigned int> iters;
    std::vector<double> times;
    for (const auto& elem : calib_root_node.get_child("niters")) {
      iters.push_back(elem.second.get<unsigned int>(""));
    }
    for (const auto& elem : calib_root_node.get_child("timesInMicroSeconds")) {
      times.push_back(elem.second.get<double>(""));
    }
    cpuCruncher_ = std::make_unique<cms::cudatest::TimeCruncher>(iters, times);
  }

  edm::Service<CUDAService> cs;
  if (cs.isAvailable() and cs->enabled()) {
    boost::property_tree::ptree calib_root_node;
    boost::property_tree::read_json(iConfig.getParameter<edm::FileInPath>("cudaCalibration").fullPath(),
                                    calib_root_node);
    std::vector<unsigned int> iters;
    std::vector<double> times;
    for (const auto& elem : calib_root_node.get_child("niters")) {
      iters.push_back(elem.second.get<unsigned int>(""));
    }
    for (const auto& elem : calib_root_node.get_child("timesInMicroSeconds")) {
      times.push_back(elem.second.get<double>(""));
    }
    gpuCruncher_ = std::make_unique<cms::cudatest::GPUTimeCruncher>(iters, times);
  }

  boost::property_tree::ptree root_node;
  boost::property_tree::read_json(iConfig.getParameter<edm::FileInPath>("config").fullPath(), root_node);

  // TODO: make gangNum per-module configurable again?
  const auto gangKernelFactor = iConfig.getParameter<double>("gangKernelFactor");
  const auto fakeUseLocks = iConfig.getParameter<bool>("fakeUseLocks");

  for (const auto& module : root_node.get_child("moduleDefinitions")) {
    const auto& moduleName = module.first;
    for (const auto& modfunc : module.second) {
      OpVectorCPU opsCPU;
      OpVectorGPU opsGPU;
      for (const auto& op : modfunc.second.get_child("")) {
        auto opname = op.second.get<std::string>("name");
        if (opname == "cpu") {
          opsCPU.emplace_back(std::make_unique<cms::cudatest::OperationCPU>(op.second, cpuCruncher_.get()));
        } else if (opname == "sleep") {
          opsCPU.emplace_back(std::make_unique<OperationSleep>(op.second, cpuCruncher_.get()));
        } else if (opname == "kernel") {
          opsGPU.emplace_back(std::make_unique<OperationKernel>(op.second, gpuCruncher_.get(), gangKernelFactor));
        } else if (opname == "memcpyHtoD") {
          opsGPU.emplace_back(std::make_unique<OperationMemcpyToDevice>(op.second));
        } else if (opname == "memcpyDtoH") {
          opsGPU.emplace_back(std::make_unique<OperationMemcpyToHost>(op.second));
        } else if (opname == "memset") {
          opsGPU.emplace_back(std::make_unique<OperationMemset>(op.second));
        } else if (opname == "fake") {
          opsGPU.emplace_back(std::make_unique<OperationFake>(op.second, cpuCruncher_.get(), fakeUseLocks));
        } else {
          throw cms::Exception("Configuration") << "Unsupported operation " << opname;
        }
      }

      size_t nevents = 0;
      if (not opsCPU.empty()) {
        nevents = opsCPU.front()->size();
        for (const auto& op : opsCPU) {
          if (op->size() != nevents) {
            throw cms::Exception("Configuration") << "Inconsistency in number of events in CPU operations of module "
                                                  << moduleName << " " << op->size() << " != " << nevents;
          }
        }
      }
      if (not opsGPU.empty()) {
        if (nevents == 0) {
          nevents = opsGPU.front()->size();
        }
        for (const auto& op : opsGPU) {
          if (op->size() != nevents) {
            throw cms::Exception("Configuration") << "Inconsistency in number of events in GPU operations of module "
                                                  << moduleName << " " << op->size() << " != " << nevents;
          }
        }
      }

      if (modfunc.first == "acquire") {
        acquireOpsCPU_.emplace_back(moduleName, std::move(opsCPU));
        acquireOpsGPU_.emplace_back(moduleName, std::move(opsGPU));
      } else if (modfunc.first == "produce") {
        produceOpsCPU_.emplace_back(moduleName, std::move(opsCPU));
        produceOpsGPU_.emplace_back(moduleName, std::move(opsGPU));
      } else {
        throw cms::Exception("Configuration") << "Unsupported module function " << modfunc.first;
      }
    }
  }

  // Initialize CPU cruncher
  cms::cudatest::getTimeCruncher();

  // TODO: fix the following
  //edm::LogWarning("foo") << "SimOperations initialized with " << ops_.size() << " operations for " << events() << " events, dropped " << ignored << " because of inconsistent configuration";
}
SimOperationsService::~SimOperationsService() {}

void SimOperationsService::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<unsigned int>("gangSize", 1);
  desc.add<unsigned int>("gangNumber", 1);
  desc.add<double>("gangKernelFactor", 1.0);
  desc.add<bool>("fakeUseLocks", true);
  desc.add<int>("maxEvents", -1);

  desc.add<edm::FileInPath>("config", edm::FileInPath())
      ->setComment("Path to a JSON configuration file of the simulation");
  desc.add<edm::FileInPath>("cpuCalibration", edm::FileInPath())
      ->setComment("Path to a JSON file for the CPU calibration");
  desc.add<edm::FileInPath>("cudaCalibration", edm::FileInPath())
      ->setComment("Path to a JSON file for the CUDA calibration");

  descriptions.addWithDefaultLabel(desc);
}

SimOperationsService::GPUData::GPUData(const OpVectorGPU& ops, const unsigned int gangSize) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> disf(1e-5, 100.);
  std::uniform_int_distribution<char> disc(-100, 100);

  // Data for kernel
  {
    constexpr auto elements = cms::cudatest::GPUTimeCruncher::kernel_elements;
    auto h_src =
        cms::cuda::make_host_noncached_unique<char[]>(elements * sizeof(float) /*, cudaHostAllocWriteCombined*/);
    cudaCheck(cudaMalloc(&kernel_data_d_, elements * sizeof(float)));
    for (size_t i = 0; i != elements; ++i) {
      h_src[i] = disf(gen);
    }
    cudaCheck(cudaMemcpy(kernel_data_d_, h_src.get(), elements * sizeof(float), cudaMemcpyDefault));
  }

  // Data for transfer operations
  data_d_src_.resize(ops.size(), nullptr);
  data_h_src_.resize(ops.size());
  for (size_t i = 0; i != ops.size(); ++i) {
    if (const auto bytesToD = gangSize * ops[i]->maxBytesToDevice(); bytesToD > 0) {
      data_h_src_[i] = cms::cuda::make_host_noncached_unique<char[]>(bytesToD /*, cudaHostAllocWriteCombined*/);
      LogTrace("foo") << "Host ptr " << i << " bytes " << bytesToD << " ptr "
                      << static_cast<const void*>(data_h_src_[i].get());
      for (unsigned int j = 0; j < bytesToD; ++j) {
        data_h_src_[i][j] = disc(gen);
      }
    }
    if (const auto bytesToH = gangSize * ops[i]->maxBytesToHost(); bytesToH > 0) {
      cudaCheck(cudaMalloc(&data_d_src_[i], bytesToH));
      LogTrace("foo") << "Device ptr " << i << " bytes " << bytesToH << " ptr "
                      << static_cast<const void*>(data_d_src_[i]);
      auto h_src = cms::cuda::make_host_noncached_unique<char[]>(bytesToH /*, cudaHostAllocWriteCombined*/);
      for (unsigned int j = 0; j < bytesToH; ++j) {
        h_src[j] = disc(gen);
      }
      cudaCheck(cudaMemcpy(data_d_src_[i], h_src.get(), bytesToH, cudaMemcpyDefault));
    }
  }
}

SimOperationsService::GPUData::~GPUData() {
  if (kernel_data_d_ != nullptr) {
    cudaCheck(cudaFree(kernel_data_d_));
    for (auto& ptr : data_d_src_) {
      cudaCheck(cudaFree(ptr));
    }
  }
}

void SimOperationsService::GPUData::swap(GPUData& rhs) {
  std::swap(kernel_data_d_, rhs.kernel_data_d_);
  std::swap(data_d_src_, rhs.data_d_src_);
  std::swap(data_h_src_, rhs.data_h_src_);
}

SimOperationsService::GPUData::GPUData(GPUData&& rhs) { swap(rhs); }

SimOperationsService::GPUData& SimOperationsService::GPUData::operator=(GPUData&& rhs) {
  GPUData tmp;
  swap(tmp);
  swap(rhs);
  return *this;
}

cms::cudatest::OperationState SimOperationsService::GPUData::makeState() {
  return cms::cudatest::OperationState{kernel_data_d_, data_h_src_, data_d_src_};
}

void SimOperationsService::AcquireGPUProcessor::process(const std::vector<size_t>& indices, cudaStream_t stream) {
  if (index_ >= 0) {
    cms::cudatest::OperationState state = data_.makeState();
    sos_->acquireGPU(index_, indices, state, stream);
  }
}

void SimOperationsService::ProduceGPUProcessor::process(const std::vector<size_t>& indices, cudaStream_t stream) {
  if (index_ >= 0) {
    cms::cudatest::OperationState state = data_.makeState();
    sos_->produceGPU(index_, indices, state, stream);
  }
}

namespace {
  template <typename T>
  std::tuple<int, size_t> indexEvents(const T& ops, const std::string& moduleLabel) {
    int index = -1;
    size_t nevents = 0;
    const auto found =
        std::find_if(ops.begin(), ops.end(), [&](const auto& elem) { return elem.first == moduleLabel; });
    if (found != ops.end()) {
      index = std::distance(ops.begin(), found);
      if (not ops[index].second.empty()) {
        nevents = ops[index].second.front()->size();
      }
    }
    return std::make_tuple(index, nevents);
  }
}  // namespace

SimOperationsService::AcquireCPUProcessor SimOperationsService::acquireCPUProcessor(
    const std::string& moduleLabel) const {
  const auto ie = indexEvents(acquireOpsCPU_, moduleLabel);
  return AcquireCPUProcessor(std::get<0>(ie), this, std::get<1>(ie));
}
SimOperationsService::AcquireGPUProcessor SimOperationsService::acquireGPUProcessor(
    const std::string& moduleLabel) const {
  return acquireGPUProcessor(moduleLabel, gangSize_);
}
SimOperationsService::AcquireGPUProcessor SimOperationsService::acquireGPUProcessor(const std::string& moduleLabel,
                                                                                    int gangSize) const {
  const auto ie = indexEvents(acquireOpsGPU_, moduleLabel);
  if (std::get<0>(ie) >= 0) {
    return AcquireGPUProcessor(
        std::get<0>(ie), this, GPUData(acquireOpsGPU_.at(std::get<0>(ie)).second, gangSize), std::get<1>(ie));
  }
  return AcquireGPUProcessor();
}
SimOperationsService::ProduceCPUProcessor SimOperationsService::produceCPUProcessor(
    const std::string& moduleLabel) const {
  const auto ie = indexEvents(produceOpsCPU_, moduleLabel);
  return ProduceCPUProcessor(std::get<0>(ie), this, std::get<1>(ie));
}
SimOperationsService::ProduceGPUProcessor SimOperationsService::produceGPUProcessor(
    const std::string& moduleLabel) const {
  const auto ie = indexEvents(produceOpsGPU_, moduleLabel);
  if (std::get<0>(ie) >= 0) {
    return ProduceGPUProcessor(
        std::get<0>(ie), this, GPUData(produceOpsGPU_.at(std::get<0>(ie)).second, gangSize_), std::get<1>(ie));
  }
  return ProduceGPUProcessor();
}

void SimOperationsService::acquireCPU(int modIndex, const std::vector<size_t>& indices) const {
  for (const auto& op : acquireOpsCPU_.at(modIndex).second) {
    op->operate(indices);
  }
}
void SimOperationsService::acquireCPU(int modIndex,
                                      const std::vector<size_t>& indices,
                                      const cms::cudatest::SleepFunction& sleep) const {
  for (const auto& op : acquireOpsCPU_.at(modIndex).second) {
    op->operate(indices, sleep);
  }
}
void SimOperationsService::acquireGPU(int modIndex,
                                      const std::vector<size_t>& indices,
                                      cms::cudatest::OperationState& state,
                                      cudaStream_t stream) const {
  for (const auto& op : acquireOpsGPU_.at(modIndex).second) {
    op->operate(indices, state, stream);
    state.opIndex++;
  }
}
void SimOperationsService::produceCPU(int modIndex, const std::vector<size_t>& indices) const {
  for (const auto& op : produceOpsCPU_.at(modIndex).second) {
    op->operate(indices);
  }
}
void SimOperationsService::produceGPU(int modIndex,
                                      const std::vector<size_t>& indices,
                                      cms::cudatest::OperationState& state,
                                      cudaStream_t stream) const {
  for (const auto& op : produceOpsGPU_.at(modIndex).second) {
    op->operate(indices, state, stream);
    state.opIndex++;
  }
}

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
DEFINE_FWK_SERVICE(SimOperationsService);
