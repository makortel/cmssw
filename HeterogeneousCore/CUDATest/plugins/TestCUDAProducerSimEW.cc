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
#include "TestCUDAProducerSimEWGPUKernel.h"

#include <random>

namespace {
  struct State {
    float* kernel_data;
    size_t kernel_elements;

    std::vector<char *> data_h_src;
    std::vector<char *> data_d_src;

    std::vector<cudautils::host::unique_ptr<char[]>> data_h_dst;
    std::vector<cudautils::host::unique_ptr<char[]>> data_d_dst;
  };

  class OperationBase {
  public:
    OperationBase() = default;
    virtual ~OperationBase() = default;

    virtual unsigned int bytesToDevice() const { return 0; }
    virtual unsigned int bytesToHost() const { return 0; }

    virtual void operate(State& state, cuda::stream_t<> *stream) const = 0;

  private:
  };

  class OperationCPU: public OperationBase {
  public:
    OperationCPU(const edm::ParameterSet& iConfig):
      time_{iConfig.getParameter<unsigned long int>("time")}
    {}

    void operate(State& state, cuda::stream_t<> *stream) const override {
      cudatest::getTimeCruncher().crunch_for(time_);
    };
  private:
    const std::chrono::nanoseconds time_;
  };

  class OperationKernel: public OperationBase {
    OperationCPU(const edm::ParameterSet& iConfig):
      time_{iConfig.getParameter<unsigned long int>("time"))}
    {}

    void operate(State& state, cuda::stream_t<> *stream) const override {
      // ???
    };
  private:
    const std::chrono::nanoseconds time_;
  }

  class OperationMemcpyToDevice: public OperationBase {
  public:
    OperationMemcpyToDevice(const edm::ParameterSet& iConfig):
      bytes_{iConfig.getParameter<unsigned int>("bytes")}
    {}

    unsigned int bytesToDevice() const override { return bytes_; }

    void operate(State& state, cuda::stream_t<> *stream) const override {
      const auto i = state.data_d_dst.size();

      auto data_d = cudautils::make_device_unique<char[]>(bytes, *stream);
      cuda::memory::async::copy(data_d.get(), state.data_h_src[i], bytes_, stream->id());
      state.data_d_dst.emplace_back(data_d);
    }

  private:
    const unsigned int bytes_;
  };

  class OperationMemcpyToHost: public OperationBase {
  public:
    OperationMemcpyToHost(const edm::ParameterSet& iConfig):
      bytes_{iConfig.getParameter<unsigned int>("bytes")}
    {}

    unsigned int bytesToHost() const override { return bytes_; }

    void operate(State& state, cuda::stream_t<> *stream) const override {
      const auto i = state.data_h_dst.size();

      auto data_d = cudautils::make_host_unique<char[]>(bytes, *stream);
      cuda::memory::async::copy(data_h.get(), state.data_d_src[i], bytes_, stream->id());
      state.data_h_dst.emplace_back(data_h);
    }

  private:
    const unsigned int bytes_;
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

  using OpsPerEventVector = std::vector<std::unique_ptr<OperationBase> >;
  using OpVector = std::vector<OpsPerEventVector>;

  OpVector acquireOps_;
  OpVector produceOps_;

  static constexpr size_t kernel_elements_ = 32;
  float* kernel_data_d_;

  std::vector<float* > data_d_src_;
  std::vector<cudautils::host::noncached::unique_ptr<float[]>> data_h_src_;
};

TestCUDAProducerSimEW::TestCUDAProducerSimEW(const edm::ParameterSet& iConfig):
  dstToken_{produces<int>()},
{
  auto createOps = [&](const std::string& psetName) {
    OpVector ret;
    for(const auto& ps: iConfig.getParameter<std::vector<edm::ParameterSet> >(psetName)) {
      ret.emplace();
      auto& opsPerEvent = ret.back();
      for(const auto& psOp: ps.getParameter<std::vector<edm::ParameterSet> >("event")) {
        auto opname = psOp.getParameter<std::string>("name");
        if(opname == "cpu") {
          opsPerEvent.emplace_back(std::make_unique<OperationCPU>(psOp));
        }
        else if(opname == "kernel") {
          opsPerEvent.emplace_back(std::make_unique<OperationKernel>(psOp));
        }
        else if(opname == "memcpyHtoD") {
          opsPerEvent.emplace_back(std::make_unique<OperationMemcpyToDevice>(psOp));
        }
        else if(opname == "memcpyDtoH") {
          opsPerEvent.emplace_back(std::make_unique<OperationMemcpyToHost>(psOp));
        }
      }
    }
  };


  edm::Service<CUDAService> cs;
  if(cs->enabled()) {
    cuda::throw_if_error(cudaMalloc(&kernel_data_d_, kernel_elements_*sizeof(float)));

    

    numberOfElementsAlloc_.resize(std::max(numberOfElementsToDevice_.size(), numberOfElementsToHost_.size()));
    data_h_.resize(numberOfElementsAlloc_.size());
    for(size_t i=0; i<data_h_.size(); ++i) {
      unsigned int elem = 0;
      if(i < numberOfElementsToDevice_.size()) {
        elem = numberOfElementsToDevice_[i];
      }
      if(i < numberOfElementsToHost_.size()) {
        elem = std::max(elem, numberOfElementsToDevice_[i]);
      }

      data_h_[i] = cudautils::make_host_noncached_unique<float[]>(elem /*, cudaHostAllocWriteCombined*/);
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1e-5, 100.);
    for(size_t i=0; i<numberOfElementsAlloc_.size(); ++i) {
      for(size_t j=0; j<numberOfElementsAlloc_[i]; ++j) {
        data_h_[i][j] = dis(gen);
      }
    }

    if(not kernelLoops_.empty()) {
      if(numberOfElementsAlloc_.empty()) {
        numberOfElementsAlloc_.resize(32);
      }
      else if(numberOfElementsAlloc_[0] < 32) {
        numberOfElementsAlloc_[0] = 32;
      }
    }

    if(not useCachingAllocator_) {
      data_d_.resize(numberOfElementsAlloc_.size());
      for(size_t i=0; i<data_d_.size(); ++i) {
        cuda::throw_if_error(cudaMalloc(&data_d_[i], numberOfElementsAlloc_[i]*sizeof(float)));
      }
    }
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
  desc.add<std::vector<edm::InputTag>> ("srcs", std::vector<edm::InputTag>{});


  edm::ParameterSetDescription opValidator;
  opValidator.add("name", std::string("cpu"));
  opValidator.addNode( edm::ParameterDescription<unsigned long int>("time", 0., true) xor
                       edm::ParameterDescription<unsigned int>("bytes", 0, true) );

  edm::ParameterSetDescription eventValidator;
  eventValidator.addVPSet("event", opValidator, std::vector<edm::ParameterSet>{});

  desc.addVPSet("acquire", eventValidator, std::vector<edm::ParameterSet>{});
  desc.addVPSet("produce", eventValidator, std::vector<edm::ParameterSet>{});

  desc.add<bool>("useCachingAllocator", true);
  descriptions.addWithDefaultLabel(desc);
}

void TestCUDAProducerSimEW::acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder h) {
  // to make sure the dependencies are set correctly
  for(const auto& t: srcTokens_) {
    iEvent.get(t);
  }

  if(crunchForMicroSeconds_.count() > 0) {
    cudatest::getTimeCruncher().crunch_for(crunchForMicroSeconds_);
  }

  CUDAScopedContextAcquire ctx{iEvent.streamID(), std::move(h)};
  std::vector<float *> data_d = data_d;
  std::vector<cudautils::device::unique_ptr<float[]>> data_d_own;
  if(useCachingAllocator_) {
    data_d.resize(numberOfElementsAlloc_.size());
    data_d_own.resize(numberOfElementsAlloc_.size());
    for(size_t i=0; i<numberOfElementsAlloc_.size(); ++i) {
      data_d_own[i] = cudautils::make_device_unique<float[]>(numberOfElementsAlloc_[i], ctx.stream());
      data_d[i] = data_d_own[i].get();
    }
  }
  for(size_t i=0; i<numberOfElementsToDevice_.size(); ++i) {
    cuda::memory::async::copy(data_d[i], data_h_[i].get(), numberOfElementsToDevice_[i]*sizeof(float), ctx.stream().id());
  }
  for(size_t i=0; i<kernelLoops_.size(); ++i) {
    TestCUDAProducerSimEWGPUKernel::kernel(data_d[0], numberOfElementsToDevice_[0], kernelLoops_[i], ctx.stream());
  }
  for(size_t i=0; i<numberOfElementsToHost_.size(); ++i) {
    cuda::memory::async::copy(data_h_[i].get(), data_d[i], numberOfElementsToHost_[i]*sizeof(float), ctx.stream().id());
  }
}

void TestCUDAProducerSimEW::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  iEvent.emplace(dstToken_, 42);
}

DEFINE_FWK_MODULE(TestCUDAProducerSimEW);
