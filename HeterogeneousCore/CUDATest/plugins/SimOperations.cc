#include "SimOperations.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_noncached_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

#include "TimeCruncher.h"
#include "GPUTimeCruncher.h"

#include <random>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>


namespace {
  struct State {
    State(const cudatest::GPUTimeCruncher* gtc,  float* kernel_data, std::vector<cudautils::host::noncached::unique_ptr<char[]>>& h_src, std::vector<char *> d_src):
      gpuCruncher{gtc},
      kernel_data_d{kernel_data},
      data_d_src{std::move(d_src)}
    {
      data_h_src.resize(h_src.size());
      std::transform(h_src.begin(), h_src.end(), data_h_src.begin(), [](auto& ptr) {
          return ptr.get();
        });
    }

    const cudatest::GPUTimeCruncher* gpuCruncher;

    float* kernel_data_d;

    std::vector<char *> data_h_src;
    std::vector<char *> data_d_src;

    std::vector<cudautils::host::unique_ptr<char[]>> data_h_dst;
    std::vector<cudautils::device::unique_ptr<char[]>> data_d_dst;

    int opIndex = 0;
  };
}

namespace cudatest {
  class OperationBase {
  public:
    OperationBase() = default;
    virtual ~OperationBase() = default;

    virtual bool emplace_back(const boost::property_tree::ptree& conf) = 0;
    virtual void pop_back() = 0;
    virtual size_t size() const = 0;

    virtual unsigned int maxBytesToDevice() const { return 0; }
    virtual unsigned int maxBytesToHost() const { return 0; }

    virtual void operate(const std::vector<size_t>& indices, State& state, cuda::stream_t<> *stream) const = 0;
  };
}

namespace {
  class OperationTime: public cudatest::OperationBase {
  public:
    explicit OperationTime(const boost::property_tree::ptree& conf)
    {
      time_.emplace_back(conf.get<unsigned long long>("time"));
    }

    bool emplace_back(const boost::property_tree::ptree& conf) override {
      if(not checkName(conf.get<std::string>("name"))) {
        return false;
      }

      time_.emplace_back(conf.get<unsigned long long>("time"));
      return true;
    }

    void pop_back() override {
      time_.pop_back();
    }

    size_t size() const override {
      return time_.size();
    }

  protected:
    std::chrono::nanoseconds totalTime(const std::vector<size_t>& indices) const {
      std::chrono::nanoseconds total{};
      for(size_t i: indices) {
        total += time_[i];
      }
      return total;
    }

    virtual bool checkName(const std::string& name) const = 0;

  private:
    std::vector<std::chrono::nanoseconds> time_;
  };

  class OperationCPU final: public OperationTime {
  public:
    explicit OperationCPU(const boost::property_tree::ptree& conf): OperationTime(conf) {}

    bool checkName(const std::string& name) const override {
      return name == "cpu";
    }

    void operate(const std::vector<size_t>& indices, State& state, cuda::stream_t<> *stream) const override {
      cudatest::getTimeCruncher().crunch_for(totalTime(indices));
    };
  };

  class OperationKernel final: public OperationTime {
  public:
    explicit OperationKernel(const boost::property_tree::ptree& conf): OperationTime(conf) {}

    bool checkName(const std::string& name) const override{
      return name == "kernel";
    }

    void operate(const std::vector<size_t>& indices, State& state, cuda::stream_t<> *stream) const override {
      assert(stream != nullptr);
      state.gpuCruncher->crunch_for(totalTime(indices), state.kernel_data_d, *stream);
    };
  };

  class OperationBytes: public cudatest::OperationBase {
  public:
    explicit OperationBytes(const boost::property_tree::ptree& conf)
    {
      bytes_.emplace_back(conf.get<unsigned int>("bytes"));
    }

    bool emplace_back(const boost::property_tree::ptree& conf) override {
      if(not checkName(conf.get<std::string>("name"))) {
        return false;
      }

      bytes_.emplace_back(conf.get<unsigned int>("bytes"));
      return true;
    }

    void pop_back() override {
      bytes_.pop_back();
    }

    size_t size() const override {
      return bytes_.size();
    }

  protected:
    unsigned long int totalBytes(const std::vector<size_t>& indices) const {
      unsigned long total = 0;
      for(size_t i: indices) {
        total += bytes_[i];
      }
      return total;
    }

    unsigned long int maxBytes() const {
      auto maxel = std::max_element(bytes_.begin(), bytes_.end());
      if(maxel == bytes_.end())
        return 0;
      return *maxel;
    }

    virtual bool checkName(const std::string& name) const = 0;

  private:
    std::vector<unsigned long int> bytes_;
  };  

  class OperationMemcpyToDevice final: public OperationBytes {
  public:
    explicit OperationMemcpyToDevice(const boost::property_tree::ptree& conf): OperationBytes(conf) {}

    unsigned int maxBytesToDevice() const override { return maxBytes(); }

    bool checkName(const std::string& name) const override {
      return name == "memcpyHtoD";
    }

    void operate(const std::vector<size_t>& indices, State& state, cuda::stream_t<> *stream) const override {
      const auto i = state.opIndex;

      const auto bytes = totalBytes(indices);

      auto data_d = cudautils::make_device_unique<char[]>(bytes, *stream);
      LogTrace("foo") << "MemcpyToDevice " << i << " from 0x" << static_cast<const void*>(state.data_h_src[i]) << " to " << static_cast<const void*>(data_d.get()) << " " << bytes << " bytes";
      cuda::memory::async::copy(data_d.get(), state.data_h_src[i], bytes, stream->id());
      state.data_d_dst.emplace_back(std::move(data_d));
    }
  };

  class OperationMemcpyToHost final: public OperationBytes {
  public:
    explicit OperationMemcpyToHost(const boost::property_tree::ptree& conf): OperationBytes(conf) {}

    unsigned int maxBytesToHost() const override { return maxBytes(); }

    bool checkName(const std::string& name) const override {
      return name == "memcpyDtoH";
    }

    void operate(const std::vector<size_t>& indices, State& state, cuda::stream_t<> *stream) const override {
      const auto i = state.opIndex;

      const auto bytes = totalBytes(indices);

      auto data_h = cudautils::make_host_unique<char[]>(bytes, *stream);
      LogTrace("foo") << "MemcpyToHost " << i << " from 0x" << static_cast<const void*>(state.data_d_src[i]) << " to " << static_cast<const void*>(data_h.get()) << " " << bytes << " bytes";
      cuda::memory::async::copy(data_h.get(), state.data_d_src[i], bytes, stream->id());
      state.data_h_dst.emplace_back(std::move(data_h));
    }
  };
}


namespace cudatest {
  SimOperations::SimOperations(const std::string& configFile,
                               const std::string& cudaCalibrationFile,
                               const std::string& nodepath,
                               const unsigned int gangSize) {
    boost::property_tree::ptree root_node;
    boost::property_tree::read_json(configFile, root_node);
    const auto& modfunc_node = root_node.get_child(nodepath);

    bool first = true;
    for(const auto& ev: modfunc_node) {
      if(first) {
        for(const auto& op: ev.second.get_child("")) {
          auto opname = op.second.get<std::string>("name");
          if(opname == "cpu") {
            ops_.emplace_back(std::make_unique<OperationCPU>(op.second));
          }
          else if(opname == "kernel") {
            ops_.emplace_back(std::make_unique<OperationKernel>(op.second));
          }
          else if(opname == "memcpyHtoD") {
            ops_.emplace_back(std::make_unique<OperationMemcpyToDevice>(op.second));
          }
          else if(opname == "memcpyDtoH") {
            ops_.emplace_back(std::make_unique<OperationMemcpyToHost>(op.second));
          }
        }
        first = false;
      }
      else {
        const auto& ops = ev.second.get_child("");
        // If different number of operations, skip the event
        if(ops.size() != ops_.size()) {
          continue;
        }

        int iOp=0;
        for(const auto& op: ops) {
          if(not ops_[iOp]->emplace_back(op.second)) {
            // If insertion failed (because of wrong type of operation), roll back the added operations from this ps
            for(--iOp; iOp >= 0; --iOp) {
              ops_[iOp]->pop_back();
            }
            break;
          }
          ++iOp;
        }
      }
    }

    // sanity check
    if(not ops_.empty()) {
      const auto nevs = ops_[0]->size();
      for(const auto& op: ops_) {
        if(op->size() != nevs) {
          throw cms::Exception("Configuration") << "Incorrect number of events " << op->size() << " vs " << nevs;
        }
      }
    }

    LogDebug("foo") << "Configured with " << ops_.size() << " operations for " << events() << " events";

    // Initialize GPU stuff
    edm::Service<CUDAService> cs;
    if(cs->enabled()) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<float> disf(1e-5, 100.);
      std::uniform_int_distribution<char> disc(-100, 100);

      // Data for kernel
      {
        constexpr auto elements = cudatest::GPUTimeCruncher::kernel_elements;
        auto h_src = cudautils::make_host_noncached_unique<char[]>(elements*sizeof(float) /*, cudaHostAllocWriteCombined*/);
        cuda::throw_if_error(cudaMalloc(&kernel_data_d_, elements*sizeof(float)));
        for(size_t i=0; i!=elements; ++i) {
          h_src[i] = disf(gen);
        }
        cuda::throw_if_error(cudaMemcpy(kernel_data_d_, h_src.get(), elements*sizeof(float), cudaMemcpyDefault));
      }
      
      // Data for transfer operations
      data_d_src_.resize(ops_.size(), nullptr);
      data_h_src_.resize(ops_.size());
      for(size_t i=0; i!=ops_.size(); ++i) {
        if(const auto bytesToD = gangSize*ops_[i]->maxBytesToDevice();
           bytesToD > 0) {
          data_h_src_[i] = cudautils::make_host_noncached_unique<char[]>(bytesToD /*, cudaHostAllocWriteCombined*/);
          LogTrace("foo") << "Host ptr " << i << " bytes " << bytesToD << " ptr " << static_cast<const void*>(data_h_src_[i].get());
          for(unsigned int j=0; j<bytesToD; ++j) {
            data_h_src_[i][j] = disc(gen);
          }
        }
        if(const auto bytesToH = gangSize*ops_[i]->maxBytesToHost();
           bytesToH > 0) {
          cuda::throw_if_error(cudaMalloc(&data_d_src_[i], bytesToH));
          LogTrace("foo") << "Device ptr " << i << " bytes " << bytesToH << " ptr " << static_cast<const void*>(data_d_src_[i]);
          auto h_src = cudautils::make_host_noncached_unique<char[]>(bytesToH /*, cudaHostAllocWriteCombined*/);
          for(unsigned int j=0; j<bytesToH; ++j) {
            h_src[j] = disc(gen);
          }
          cuda::throw_if_error(cudaMemcpy(data_d_src_[i], h_src.get(), bytesToH, cudaMemcpyDefault));
        }
      }
      boost::property_tree::ptree calib_root_node;
      boost::property_tree::read_json(cudaCalibrationFile, calib_root_node);
      std::vector<unsigned int> iters;
      std::vector<double> times;
      for(const auto& elem: calib_root_node.get_child("niters")) {
        iters.push_back(elem.second.get<unsigned int>(""));
      }
      for(const auto& elem: calib_root_node.get_child("timesInMicroSeconds")) {
        times.push_back(elem.second.get<double>(""));
      }

      gpuCruncher_ = std::make_unique<GPUTimeCruncher>(iters, times);
    }

    // Initialize CPU cruncher
    cudatest::getTimeCruncher();
  }

  SimOperations::~SimOperations() {
    cuda::throw_if_error(cudaFree(kernel_data_d_));
    for(auto& ptr: data_d_src_) {
      cuda::throw_if_error(cudaFree(ptr));
    }
  }

  size_t SimOperations::events() const {
    if(ops_.empty())
      return 0;
    return ops_[0]->size();
  }

  void SimOperations::operate(const std::vector<size_t>& indices, cuda::stream_t<>* stream) {
    State opState{gpuCruncher_.get(), kernel_data_d_, data_h_src_, data_d_src_};
    for(auto& op: ops_) {
      op->operate(indices, opState, stream);
      opState.opIndex++;
    }
  }
}
