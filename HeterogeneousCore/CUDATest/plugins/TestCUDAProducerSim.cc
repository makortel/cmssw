#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "CUDADataFormats/Common/interface/CUDAProduct.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"

#include "SimOperations.h"

class TestCUDAProducerSim: public edm::stream::EDProducer<> {
public:
  explicit TestCUDAProducerSim(const edm::ParameterSet& iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
private:
  
  std::vector<edm::EDGetTokenT<int>> srcTokens_;
  std::vector<edm::EDGetTokenT<CUDAProduct<int>>> cudaSrcTokens_;
  edm::EDPutTokenT<int> dstToken_;
  edm::EDPutTokenT<CUDAProduct<int>> cudaDstToken_;

  cudatest::SimOperations produceOps_;
};

TestCUDAProducerSim::TestCUDAProducerSim(const edm::ParameterSet& iConfig):
  produceOps_{iConfig.getParameter<edm::FileInPath>("config").fullPath(),
              iConfig.getParameter<edm::FileInPath>("cudaCalibration").fullPath(),
              "moduleDefinitions."+iConfig.getParameter<std::string>("@module_label")+".produce"}
{
  if(produceOps_.events() == 0) {
    throw cms::Exception("Configuration") << "Got 0 events, which makes this module useless";
  }

  for(const auto& src: iConfig.getParameter<std::vector<edm::InputTag>>("srcs")) {
    srcTokens_.emplace_back(consumes<int>(src));
  }
  for(const auto& src: iConfig.getParameter<std::vector<edm::InputTag>>("cudaSrcs")) {
    cudaSrcTokens_.emplace_back(consumes<CUDAProduct<int>>(src));
  }

  if(iConfig.getParameter<bool>("produce")) {
    dstToken_ = produces<int>();
  }
  if(iConfig.getParameter<bool>("produceCUDA")) {
    cudaDstToken_ = produces<CUDAProduct<int>>();
  }
}

void TestCUDAProducerSim::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("srcs", std::vector<edm::InputTag>{});
  desc.add<std::vector<edm::InputTag>>("cudaSrcs", std::vector<edm::InputTag>{});
  desc.add<bool>("produce", false);
  desc.add<bool>("produceCUDA", false);

  desc.add<edm::FileInPath>("config", edm::FileInPath())->setComment("Path to a JSON configuration file of the simulation");
  desc.add<edm::FileInPath>("cudaCalibration", edm::FileInPath())->setComment("Path to a JSON file for the CUDA calibration");

  //desc.add<bool>("useCachingAllocator", true);
  descriptions.addWithDefaultLabel(desc);
}

void TestCUDAProducerSim::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // to make sure the dependencies are set correctly
  for(const auto& t: srcTokens_) {
    iEvent.get(t);
  }

  std::vector<const CUDAProduct<int> *> cudaProducts(cudaSrcTokens_.size(), nullptr);
  std::transform(cudaSrcTokens_.begin(), cudaSrcTokens_.end(), cudaProducts.begin(), [&iEvent](const auto& tok) {
      return &iEvent.get(tok);
    });

  auto ctx = cudaProducts.empty() ? CUDAScopedContextProduce(iEvent.streamID()) :
    CUDAScopedContextProduce(*cudaProducts[0]);

  for(const auto ptr: cudaProducts) {
    ctx.get(*ptr);
  }

  produceOps_.operate(std::vector<size_t>{iEvent.id().event() % produceOps_.events()}, &ctx.stream());

  if(not dstToken_.isUninitialized()) {
    iEvent.emplace(dstToken_, 42);
  }
  if(not cudaDstToken_.isUninitialized()) {
    ctx.emplace(iEvent, cudaDstToken_, 42);
  }
}

DEFINE_FWK_MODULE(TestCUDAProducerSim);
