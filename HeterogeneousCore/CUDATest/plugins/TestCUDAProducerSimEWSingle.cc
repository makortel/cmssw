#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"

#include "SimOperationsService.h"

class TestCUDAProducerSimEWSingle : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit TestCUDAProducerSimEWSingle(const edm::ParameterSet& iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  cms::cuda::ContextState ctxState_;

  struct Module {
    Module(SimOperationsService::AcquireCPUProcessor ac,
           SimOperationsService::AcquireGPUProcessor ag,
           SimOperationsService::ProduceCPUProcessor pc,
           SimOperationsService::ProduceGPUProcessor pg)
        : acquireOpsCPU(std::move(ac)),
          acquireOpsGPU(std::move(ag)),
          produceOpsCPU(std::move(pc)),
          produceOpsGPU(std::move(pg)) {}
    SimOperationsService::AcquireCPUProcessor acquireOpsCPU;
    SimOperationsService::AcquireGPUProcessor acquireOpsGPU;
    SimOperationsService::ProduceCPUProcessor produceOpsCPU;
    SimOperationsService::ProduceGPUProcessor produceOpsGPU;
  };

  std::vector<Module> modules_;
};

TestCUDAProducerSimEWSingle::TestCUDAProducerSimEWSingle(const edm::ParameterSet& iConfig) {
  edm::Service<SimOperationsService> sos;
  for (const auto& moduleLabel : iConfig.getParameter<std::vector<std::string>>("modules")) {
    auto acquireOpsCPU = sos->acquireCPUProcessor(moduleLabel);
    auto acquireOpsGPU = sos->acquireGPUProcessor(moduleLabel);
    auto produceOpsCPU = sos->produceCPUProcessor(moduleLabel);
    auto produceOpsGPU = sos->produceGPUProcessor(moduleLabel);

    if (acquireOpsCPU.events() == 0 && acquireOpsGPU.events() == 0) {
      throw cms::Exception("Configuration")
          << "Got 0 events for module " << moduleLabel << ", which makes this module useless";
    }
    const auto nevents = std::max(acquireOpsCPU.events(), acquireOpsGPU.events());

    if (nevents != produceOpsCPU.events() and produceOpsCPU.events() > 0) {
      throw cms::Exception("Configuration") << "Got " << nevents << " events for acquire and " << produceOpsCPU.events()
                                            << " for produce CPU for module " << moduleLabel;
    }
    if (nevents != produceOpsGPU.events() and produceOpsGPU.events() > 0) {
      throw cms::Exception("Configuration") << "Got " << nevents << " events for acquire and " << produceOpsGPU.events()
                                            << " for produce GPU for module " << moduleLabel;
    }

    modules_.emplace_back(
        std::move(acquireOpsCPU), std::move(acquireOpsGPU), std::move(produceOpsCPU), std::move(produceOpsGPU));
  }
}

void TestCUDAProducerSimEWSingle::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<std::string>>("modules", std::vector<std::string>{});

  //desc.add<bool>("useCachingAllocator", true);
  descriptions.addWithDefaultLabel(desc);
}

void TestCUDAProducerSimEWSingle::acquire(const edm::Event& iEvent,
                                          const edm::EventSetup& iSetup,
                                          edm::WaitingTaskWithArenaHolder h) {
  auto ctx = cms::cuda::ScopedContextAcquire(iEvent.streamID(), std::move(h), ctxState_);

  for (auto& m : modules_) {
    if (m.acquireOpsCPU.events() > 0) {
      m.acquireOpsCPU.process(std::vector<size_t>{iEvent.id().event() % m.acquireOpsCPU.events()});
    }

    if (m.acquireOpsGPU.events() > 0) {
      m.acquireOpsGPU.process(std::vector<size_t>{iEvent.id().event() % m.acquireOpsGPU.events()}, ctx.stream());
    }
  }
}

void TestCUDAProducerSimEWSingle::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  cms::cuda::ScopedContextProduce ctx{ctxState_};

  for (auto& m : modules_) {
    if (m.produceOpsCPU.events() > 0) {
      m.produceOpsCPU.process(std::vector<size_t>{iEvent.id().event() % m.produceOpsCPU.events()});
    }

    if (m.produceOpsGPU.events() > 0) {
      m.produceOpsGPU.process(std::vector<size_t>{iEvent.id().event() % m.produceOpsGPU.events()}, ctx.stream());
    }
  }
}

DEFINE_FWK_MODULE(TestCUDAProducerSimEWSingle);
