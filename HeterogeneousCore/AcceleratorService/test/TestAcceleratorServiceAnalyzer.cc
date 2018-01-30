#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/transform.h"

#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"

#include <vector>

class TestAcceleratorServiceAnalyzer: public edm::global::EDAnalyzer<> {
public:
  explicit TestAcceleratorServiceAnalyzer(edm::ParameterSet const& iConfig);
  ~TestAcceleratorServiceAnalyzer() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::StreamID streamID, const edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  using InputType = HeterogeneousProduct<unsigned int, std::pair<float *, float *>>;
  std::string label_;
  std::vector<edm::EDGetTokenT<InputType>> srcTokens_;
};

TestAcceleratorServiceAnalyzer::TestAcceleratorServiceAnalyzer(const edm::ParameterSet& iConfig):
  label_(iConfig.getParameter<std::string>("@module_label")),
  srcTokens_(edm::vector_transform(iConfig.getParameter<std::vector<edm::InputTag> >("src"),
                                   [this](const edm::InputTag& tag) {
                                     return consumes<InputType>(tag);
                                   }))
{}

void TestAcceleratorServiceAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("src", std::vector<edm::InputTag>{});
  descriptions.add("testAcceleratorServiceAnalyzer", desc);
}

void TestAcceleratorServiceAnalyzer::analyze(edm::StreamID streamID, const edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Handle<InputType> hinput;
  int inp=0;
  for(const auto& token: srcTokens_) {
    iEvent.getByToken(token, hinput);
    edm::LogPrint("TestAcceleratorServiceAnalyzer") << "Analyzer event " << iEvent.id().event()
                                                    << " stream " << streamID
                                                    << " label " << label_
                                                    << " coll " << inp
                                                    << " result " << hinput->getCPUProduct();
    ++inp;
  }
}

DEFINE_FWK_MODULE(TestAcceleratorServiceAnalyzer);
