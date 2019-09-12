#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"
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
    State(float* kernel_data, std::vector<cudautils::host::noncached::unique_ptr<char[]>>& h_src, std::vector<char *> d_src):
      kernel_data_d{kernel_data},
      data_d_src{std::move(d_src)}
    {
      data_h_src.resize(h_src.size());
      std::transform(h_src.begin(), h_src.end(), data_h_src.begin(), [](auto& ptr) {
          return ptr.get();
        });
    }

    float* kernel_data_d;

    std::vector<char *> data_h_src;
    std::vector<char *> data_d_src;

    std::vector<cudautils::host::unique_ptr<char[]>> data_h_dst;
    std::vector<cudautils::device::unique_ptr<char[]>> data_d_dst;

    int opIndex = 0;
  };

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

  private:
  };

  class OperationTime: public OperationBase {
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
      cudatest::getGPUTimeCruncher().crunch_for(totalTime(indices), state.kernel_data_d, *stream);
    };
  };

  class OperationBytes: public OperationBase {
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
      LogTrace("foo") << "MemcpyToDevice " << i << " from 0x" << static_cast<const void*>(state.data_h_src[i]) << " to " << static_cast<const void*>(data_d.get());
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
      LogTrace("foo") << "MemcpyToHost " << i << " from 0x" << static_cast<const void*>(state.data_d_src[i]) << " to " << static_cast<const void*>(data_h.get());
      cuda::memory::async::copy(data_h.get(), state.data_d_src[i], bytes, stream->id());
      state.data_h_dst.emplace_back(std::move(data_h));
    }
  };
}


class TestCUDAProducerSimEW: public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit TestCUDAProducerSimEW(const edm::ParameterSet& iConfig);
  ~TestCUDAProducerSimEW() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
private:
  
  std::vector<edm::EDGetTokenT<int>> srcTokens_;
  const edm::EDPutTokenT<int> dstToken_;
  CUDAContextState ctxState_;

  using OpVector = std::vector<std::unique_ptr<OperationBase>>;

  OpVector acquireOps_;
  OpVector produceOps_;

  float* kernel_data_d_;

  // These are indexed by the operation index in acquireOps and produceOps_
  // They are likely to contain null elements
  std::vector<char* > data_d_src_;
  std::vector<cudautils::host::noncached::unique_ptr<char[]>> data_h_src_;
};

TestCUDAProducerSimEW::TestCUDAProducerSimEW(const edm::ParameterSet& iConfig):
  dstToken_{produces<int>()}
{

  const auto& configFileName = iConfig.getParameter<edm::FileInPath>("config");
  boost::property_tree::ptree root_node;
  boost::property_tree::read_json(configFileName.fullPath(), root_node);
  const auto& module_node = root_node.get_child("moduleDefinitions."+iConfig.getParameter<std::string>("@module_label"));
  
  auto createOps = [&](const std::string& psetName) {
    OpVector ret;
    bool first = true;
    for(const auto& ev: module_node.get_child(psetName)) {
      if(first) {
        for(const auto& op: ev.second.get_child("")) {
          auto opname = op.second.get<std::string>("name");
          if(opname == "cpu") {
            ret.emplace_back(std::make_unique<OperationCPU>(op.second));
          }
          else if(opname == "kernel") {
            ret.emplace_back(std::make_unique<OperationKernel>(op.second));
          }
          else if(opname == "memcpyHtoD") {
            ret.emplace_back(std::make_unique<OperationMemcpyToDevice>(op.second));
          }
          else if(opname == "memcpyDtoH") {
            ret.emplace_back(std::make_unique<OperationMemcpyToHost>(op.second));
          }
        }
      }
      else {
        const auto& ops = ev.second.get_child("");
        // If different number of operations, skip the event
        if(ops.size() != ret.size()) {
          continue;
        }

        int iOp=0;
        for(const auto& op: ops) {
          if(not ret[iOp]->emplace_back(op.second)) {
            // If insertion failed (because of wrong type of operation), roll back the added operations from this ps
            for(--iOp; iOp >= 0; --iOp) {
              ret[iOp]->pop_back();
            }
            break;
          }
          ++iOp;
        }
      }
    }

    // sanity check
    if(not ret.empty()) {
      const auto nevs = ret[0]->size();
      for(const auto& op: ret) {
        if(op->size() != nevs) {
          throw cms::Exception("Configuration") << "Incorrect number of events " << op->size() << " vs " << nevs;
        }
      }
    }
    return ret;
  };

  acquireOps_ = createOps("acquire");
  produceOps_ = createOps("produce");

  if((not acquireOps_.empty()) and (not produceOps_.empty()) and (acquireOps_[0]->size() != produceOps_[0]->size())) {
    throw cms::Exception("Configuration") << "Got " << acquireOps_[0]->size() << " events for acquire and " << produceOps_[0]->size() << " for produce";
  }

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
    const size_t maxops = std::max(acquireOps_.size(), produceOps_.size());

    data_d_src_.resize(maxops, nullptr);
    data_h_src_.resize(maxops);
    for(size_t i=0; i!=maxops; ++i) {
      if(const auto bytesToD = std::max(i < acquireOps_.size() ? acquireOps_[i]->maxBytesToDevice() : 0U,
                                        i < produceOps_.size() ? produceOps_[i]->maxBytesToDevice() : 0U);
         bytesToD > 0) {
        data_h_src_[i] = cudautils::make_host_noncached_unique<char[]>(bytesToD /*, cudaHostAllocWriteCombined*/);
        LogTrace("foo") << "Host ptr " << i << " bytes " << bytesToD << " ptr " << static_cast<const void*>(data_h_src_[i].get());
        for(unsigned int j=0; j<bytesToD; ++j) {
          data_h_src_[i][j] = disc(gen);
        }
      }
      if(const auto bytesToH = std::max(i < acquireOps_.size() ? acquireOps_[i]->maxBytesToHost() : 0U,
                                        i < produceOps_.size() ? produceOps_[i]->maxBytesToHost() : 0U);
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
    cudatest::getGPUTimeCruncher();
  }

  for(const auto& src: iConfig.getParameter<std::vector<edm::InputTag>>("srcs")) {
    srcTokens_.emplace_back(consumes<int>(src));
  }
  cudatest::getTimeCruncher();
}

TestCUDAProducerSimEW::~TestCUDAProducerSimEW() {
  cuda::throw_if_error(cudaFree(kernel_data_d_));
  for(auto& ptr: data_d_src_) {
    cuda::throw_if_error(cudaFree(ptr));
  }
}

void TestCUDAProducerSimEW::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("srcs", std::vector<edm::InputTag>{});

  desc.add<edm::FileInPath>("config", edm::FileInPath())->setComment("Path to a JSON configuration file of the simulation");

  //desc.add<bool>("useCachingAllocator", true);
  descriptions.addWithDefaultLabel(desc);
}

void TestCUDAProducerSimEW::acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder h) {
  // to make sure the dependencies are set correctly
  for(const auto& t: srcTokens_) {
    iEvent.get(t);
  }

  CUDAScopedContextAcquire ctx{iEvent.streamID(), std::move(h), ctxState_};

  State opState{kernel_data_d_, data_h_src_, data_d_src_};

  for(auto& op: acquireOps_) {
    op->operate(std::vector<size_t>{iEvent.id().event() % op->size()}, opState, &ctx.stream());
    opState.opIndex++;
  }
}

void TestCUDAProducerSimEW::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  CUDAScopedContextProduce ctx{ctxState_};

  State opState{kernel_data_d_, data_h_src_, data_d_src_};

  for(auto& op: produceOps_) {
    op->operate(std::vector<size_t>{iEvent.id().event() % op->size()}, opState, &ctx.stream());
    opState.opIndex++;
  }

  iEvent.emplace(dstToken_, 42);
}

DEFINE_FWK_MODULE(TestCUDAProducerSimEW);
