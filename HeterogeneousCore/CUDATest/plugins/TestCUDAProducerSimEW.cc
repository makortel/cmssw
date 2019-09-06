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

namespace {
  struct State {
    State(std::vector<cudautils::host::noncached::unique_ptr<char[]>>& h_src, std::vector<char *> d_src):
      data_d_src(std::move(d_src))
    {
      data_h_src.resize(h_src.size());
      std::transform(h_src.begin(), h_src.end(), data_h_src.begin(), [](auto& ptr) {
          return ptr.get();
        });
    }

    std::vector<char *> data_h_src;
    std::vector<char *> data_d_src;

    std::vector<cudautils::host::unique_ptr<char[]>> data_h_dst;
    std::vector<cudautils::device::unique_ptr<char[]>> data_d_dst;
  };

  class OperationBase {
  public:
    OperationBase() = default;
    virtual ~OperationBase() = default;

    virtual bool emplace_back(const edm::ParameterSet& iConfig) = 0;
    virtual void pop_back() = 0;
    virtual size_t size() const = 0;

    virtual unsigned int maxBytesToDevice() const { return 0; }
    virtual unsigned int maxBytesToHost() const { return 0; }

    virtual void operate(const std::vector<size_t>& indices, State& state, cuda::stream_t<> *stream) const = 0;

  private:
  };

  class OperationTime: public OperationBase {
  public:
    explicit OperationTime(const edm::ParameterSet& iConfig)
    {
      time_.emplace_back(iConfig.getParameter<unsigned long long>("time"));
    }

    bool emplace_back(const edm::ParameterSet& iConfig) override {
      if(not checkName(iConfig.getParameter<std::string>("name"))) {
        return false;
      }

      time_.emplace_back(iConfig.getParameter<unsigned long long>("time"));
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
    explicit OperationCPU(const edm::ParameterSet& iConfig): OperationTime(iConfig) {}

    bool checkName(const std::string& name) const override {
      return name == "cpu";
    }

    void operate(const std::vector<size_t>& indices, State& state, cuda::stream_t<> *stream) const override {
      cudatest::getTimeCruncher().crunch_for(totalTime(indices));
    };
  };

  class OperationKernel final: public OperationTime {
  public:
    explicit OperationKernel(const edm::ParameterSet& iConfig): OperationTime(iConfig) {}

    bool checkName(const std::string& name) const override{
      return name == "kernel";
    }

    void operate(const std::vector<size_t>& indices, State& state, cuda::stream_t<> *stream) const override {
      assert(stream != nullptr);
      cudatest::getGPUTimeCruncher().crunch_for(totalTime(indices), *stream);
    };
  };

  class OperationBytes: public OperationBase {
  public:
    explicit OperationBytes(const edm::ParameterSet& iConfig)
    {
      bytes_.emplace_back(iConfig.getParameter<unsigned int>("bytes"));
    }

    bool emplace_back(const edm::ParameterSet& iConfig) override {
      if(not checkName(iConfig.getParameter<std::string>("name"))) {
        return false;
      }

      bytes_.emplace_back(iConfig.getParameter<unsigned int>("bytes"));
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
    explicit OperationMemcpyToDevice(const edm::ParameterSet& iConfig): OperationBytes(iConfig) {}

    unsigned int maxBytesToDevice() const override { return maxBytes(); }

    bool checkName(const std::string& name) const override {
      return name == "memcpyHtoD";
    }

    void operate(const std::vector<size_t>& indices, State& state, cuda::stream_t<> *stream) const override {
      const auto i = state.data_d_dst.size();

      const auto bytes = totalBytes(indices);

      auto data_d = cudautils::make_device_unique<char[]>(bytes, *stream);
      cuda::memory::async::copy(data_d.get(), state.data_h_src[i], bytes, stream->id());
      state.data_d_dst.emplace_back(std::move(data_d));
    }
  };

  class OperationMemcpyToHost final: public OperationBytes {
  public:
    explicit OperationMemcpyToHost(const edm::ParameterSet& iConfig): OperationBytes(iConfig) {}

    unsigned int maxBytesToHost() const override { return maxBytes(); }

    bool checkName(const std::string& name) const override {
      return name == "memcpyDtoH";
    }

    void operate(const std::vector<size_t>& indices, State& state, cuda::stream_t<> *stream) const override {
      const auto i = state.data_h_dst.size();

      const auto bytes = totalBytes(indices);

      auto data_h = cudautils::make_host_unique<char[]>(bytes, *stream);
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

  // These are indexed by the operation index in acquireOps and produceOps_
  // They are likely to contain null elements
  std::vector<char* > data_d_src_;
  std::vector<cudautils::host::noncached::unique_ptr<char[]>> data_h_src_;
};

TestCUDAProducerSimEW::TestCUDAProducerSimEW(const edm::ParameterSet& iConfig):
  dstToken_{produces<int>()}
{
  auto createOps = [&](const std::string& psetName) {
    OpVector ret;
    bool first = true;
    for(const auto& ps: iConfig.getParameter<std::vector<edm::ParameterSet> >(psetName)) {
      if(first) {
        for(const auto& psOp: ps.getParameter<std::vector<edm::ParameterSet> >("event")) {
          auto opname = psOp.getParameter<std::string>("name");
          if(opname == "cpu") {
            ret.emplace_back(std::make_unique<OperationCPU>(psOp));
          }
          else if(opname == "kernel") {
            ret.emplace_back(std::make_unique<OperationKernel>(psOp));
          }
          else if(opname == "memcpyHtoD") {
            ret.emplace_back(std::make_unique<OperationMemcpyToDevice>(psOp));
          }
          else if(opname == "memcpyDtoH") {
            ret.emplace_back(std::make_unique<OperationMemcpyToHost>(psOp));
          }
        }
        first = false;
      }
      else {
        auto ops = ps.getParameter<std::vector<edm::ParameterSet> >("event");
        // If different number of operations, skip the event
        if(ops.size() != ret.size()) {
          continue;
        }

        int iOp=0;
        for(const auto& psOp: ops) {
          if(not ret[iOp]->emplace_back(psOp)) {
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
    std::uniform_int_distribution<char> dis(-100, 100);

    // Data for transfer operations
    const size_t maxops = std::max(acquireOps_.size(), produceOps_.size());

    data_d_src_.resize(maxops, nullptr);
    data_h_src_.resize(maxops);
    for(size_t i=0; i!=maxops; ++i) {
      if(const auto bytesToD = std::max(i < acquireOps_.size() ? acquireOps_[i]->maxBytesToDevice() : 0U,
                                        i < produceOps_.size() ? produceOps_[i]->maxBytesToDevice() : 0U);
         bytesToD > 0) {
        data_h_src_[i] = cudautils::make_host_noncached_unique<char[]>(bytesToD /*, cudaHostAllocWriteCombined*/);
        for(unsigned int j=0; j<bytesToD; ++j) {
          data_h_src_[i][j] = dis(gen);
        }
      }
      if(const auto bytesToH = std::max(i < acquireOps_.size() ? acquireOps_[i]->maxBytesToHost() : 0U,
                                        i < produceOps_.size() ? produceOps_[i]->maxBytesToHost() : 0U);
         bytesToH > 0) {
        cuda::throw_if_error(cudaMalloc(&data_d_src_[i], bytesToH));
        auto h_src = cudautils::make_host_noncached_unique<char[]>(bytesToH /*, cudaHostAllocWriteCombined*/);
        for(unsigned int j=0; j<bytesToH; ++j) {
          h_src[j] = dis(gen);
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
  for(auto& ptr: data_d_src_) {
    cuda::throw_if_error(cudaFree(ptr));
  }
}

void TestCUDAProducerSimEW::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>> ("srcs", std::vector<edm::InputTag>{});


  edm::ParameterSetDescription opValidator;
  opValidator.add("name", std::string("cpu"));
  opValidator.addNode( edm::ParameterDescription<unsigned long long>("time", 0., true) xor
                       edm::ParameterDescription<unsigned int>("bytes", 0, true) );

  edm::ParameterSetDescription eventValidator;
  eventValidator.addVPSet("event", opValidator, std::vector<edm::ParameterSet>{});

  desc.addVPSet("acquire", eventValidator, std::vector<edm::ParameterSet>{});
  desc.addVPSet("produce", eventValidator, std::vector<edm::ParameterSet>{});

  //desc.add<bool>("useCachingAllocator", true);
  descriptions.addWithDefaultLabel(desc);
}

void TestCUDAProducerSimEW::acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder h) {
  // to make sure the dependencies are set correctly
  for(const auto& t: srcTokens_) {
    iEvent.get(t);
  }

  CUDAScopedContextAcquire ctx{iEvent.streamID(), std::move(h), ctxState_};

  State opState{data_h_src_, data_d_src_};

  for(auto& op: acquireOps_) {
    op->operate(std::vector<size_t>{iEvent.id().event() % op->size()}, opState, &ctx.stream());
  }
}

void TestCUDAProducerSimEW::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  CUDAScopedContextProduce ctx{ctxState_};

  State opState{data_h_src_, data_d_src_};

  for(auto& op: produceOps_) {
    op->operate(std::vector<size_t>{iEvent.id().event() % op->size()}, opState, &ctx.stream());
  }

  iEvent.emplace(dstToken_, 42);
}

DEFINE_FWK_MODULE(TestCUDAProducerSimEW);
