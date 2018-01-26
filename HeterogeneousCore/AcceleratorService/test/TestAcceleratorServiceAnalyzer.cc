#include "FWCore/Framework/interface/global/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"

class TestAcceleratorServiceAnalyzer: public edm::global::EDAnalyzer<> {
public:
  explicit TestAcceleratorServiceAnalyzer(edm::ParameterSet const& iConfig);
  ~TestAcceleratorServiceAnalyzer() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::StreamID streamID, const edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  using InputType = HeterogeneousProduct<unsigned int, float *>;
  std::string label_;
  edm::EDGetTokenT<InputType> srcToken_;
};

TestAcceleratorServiceAnalyzer::TestAcceleratorServiceAnalyzer(const edm::ParameterSet& iConfig):
  label_(iConfig.getParameter<std::string>("@module_label")),
  srcToken_(consumes<InputType>(iConfig.getParameter<edm::InputTag>("src")))
{}

void TestAcceleratorServiceAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag());
  descriptions.add("testAcceleratorServiceAnalyzer", desc);
}

void TestAcceleratorServiceAnalyzer::analyze(edm::StreamID streamID, const edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Handle<InputType> hinput;
  iEvent.getByToken(srcToken_, hinput);
  edm::LogPrint("TestAcceleratorServiceAnalyzer") << "Analyzer event " << iEvent.id().event()
                                                  << " stream " << streamID
                                                  << " label " << label_
                                                  << " result " << hinput->getCPUProduct();
}

DEFINE_FWK_MODULE(TestAcceleratorServiceAnalyzer);
