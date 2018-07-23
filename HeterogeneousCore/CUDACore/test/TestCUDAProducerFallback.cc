#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/transform.h"

class TestCUDAProducerFallback: public edm::global::EDProducer<> {
public:
  explicit TestCUDAProducerFallback(const edm::ParameterSet& iConfig);
  ~TestCUDAProducerFallback() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const;

private:
  std::string label_;
  std::vector<edm::EDGetTokenT<int>> tokens_;
};

TestCUDAProducerFallback::TestCUDAProducerFallback(const edm::ParameterSet& iConfig):
  label_(iConfig.getParameter<std::string>("@module_label")),
  tokens_(edm::vector_transform(iConfig.getParameter<std::vector<edm::InputTag> >("src"),
                                [this](const edm::InputTag& tag) {
                                  return consumes<int>(tag);
                                }))
{
  produces<int>();
}

void TestCUDAProducerFallback::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("src", std::vector<edm::InputTag>{})->setComment("Ordered list of input 'int' inputs.");
  descriptions.addWithDefaultLabel(desc);
  descriptions.setComment("This EDProducer is part of the TestCUDAProducer* family. It acts as an enhanced EDAlias with a defined order of inputs. I.e. if first input is available, copy that. If not, try the next one etc. If no inputs are available, throw an exception. To be replaced with an EDAlias-style feature in the framework.");
}

void TestCUDAProducerFallback::produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::LogPrint("TestCUDAProducerFallback") << label_ << " TestCUDAProducerFallback::produce begin event " << iEvent.id().event() << " stream " << iEvent.streamID();
  edm::Handle<int> hin;
  for(const auto& token: tokens_) {
    edm::EDConsumerBase::Labels labels;
    labelsForToken(token, labels);
    if(iEvent.getByToken(token, hin)) {
      edm::LogPrint("TestCUDAProducerFallback") << label_ << "  input " << labels.module << " found";
      iEvent.put(std::make_unique<int>(*hin));
      return;
    }
    edm::LogPrint("TestCUDAProducerFallback") << label_ << "  input " << labels.module << " NOT found";
  }
  throw cms::Exception("ProductNotFound") << "Unable to find product 'int' from any of the inputs";
}

DEFINE_FWK_MODULE(TestCUDAProducerFallback);
