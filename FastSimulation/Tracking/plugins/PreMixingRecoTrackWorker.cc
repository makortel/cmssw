#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ProducesCollector.h"

#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include "SimGeneral/PreMixingModule/interface/PreMixingWorker.h"
#include "SimGeneral/PreMixingModule/interface/PreMixingWorkerFactory.h"

#include "RecoTrackAccumulator.h"

class PreMixingRecoTrackWorker : public PreMixingWorker {
public:
  PreMixingRecoTrackWorker(const edm::ParameterSet& ps,
                           edm::ProducesCollector producesCollector,
                           edm::ConsumesCollector&& iC)
      : accumulator_(ps, producesCollector, iC) {}

  void initializeEvent(const edm::Event& e, const edm::EventSetup& ES) override { accumulator_.initializeEvent(e, ES); }

  void initializeBunchCrossing(edm::Event const& e, edm::EventSetup const& ES, int bunchCrossing) override {
    accumulator_.initializeBunchCrossing(e, ES, bunchCrossing);
  }
  void finalizeBunchCrossing(edm::Event& e, edm::EventSetup const& ES, int bunchCrossing) override {
    accumulator_.finalizeBunchCrossing(e, ES, bunchCrossing);
  }

  void addSignals(const edm::Event& e, const edm::EventSetup& ES) override { accumulator_.accumulate(e, ES); }
  void addPileups(PileUpEventPrincipal const& pep, edm::EventSetup const& ES) override {
    accumulator_.accumulate(pep, ES, pep.principal().streamID());
  }
  void put(edm::Event& e, const edm::EventSetup& ES, std::vector<PileupSummaryInfo> const& ps, int bs) override {
    accumulator_.finalizeEvent(e, ES);
  }

private:
  RecoTrackAccumulator accumulator_;
};

DEFINE_PREMIXING_WORKER(PreMixingRecoTrackWorker);
