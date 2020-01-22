#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <mutex>

namespace {
  std::mutex mutex;
}

class TestCUDAProducerOverheadEW : public edm::global::EDProducer<edm::ExternalWork> {
public:
  explicit TestCUDAProducerOverheadEW(const edm::ParameterSet& iConfig);
  ~TestCUDAProducerOverheadEW() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void acquire(edm::StreamID id,
               const edm::Event& iEvent,
               const edm::EventSetup& iSetup,
               edm::WaitingTaskWithArenaHolder) const override;
  void produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

private:
  edm::EDPutTokenT<int> dstToken_;
  const bool lockMutex_;
};

TestCUDAProducerOverheadEW::TestCUDAProducerOverheadEW(const edm::ParameterSet& iConfig)
    : dstToken_{produces<int>()}, lockMutex_{iConfig.getParameter<bool>("lockMutex")} {}

void TestCUDAProducerOverheadEW::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("lockMutex", false);
  descriptions.addWithDefaultLabel(desc);
}

void TestCUDAProducerOverheadEW::acquire(edm::StreamID id,
                                         const edm::Event& iEvent,
                                         const edm::EventSetup& iSetup,
                                         edm::WaitingTaskWithArenaHolder) const {
  if (lockMutex_)
    std::lock_guard<std::mutex> lock{mutex};
}

void TestCUDAProducerOverheadEW::produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  iEvent.emplace(dstToken_, 42);
}

DEFINE_FWK_MODULE(TestCUDAProducerOverheadEW);
