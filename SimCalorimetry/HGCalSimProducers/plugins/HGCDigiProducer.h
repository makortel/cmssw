#ifndef SimCalorimetry_HGCSimProducers_HGCDigiProducer_h
#define SimCalorimetry_HGCSimProducers_HGCDigiProducer_h

#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"
#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizer.h"
#include "FWCore/Framework/interface/ProducesCollector.h"

#include <vector>

namespace edm {
  class ConsumesCollector;
  namespace stream {
    class EDProducerBase;
  }
  class ParameterSet;
  class StreamID;
}  // namespace edm

namespace CLHEP {
  class HepRandomEngine;
}

class HGCDigiProducer : public DigiAccumulatorMixMod {
public:
  HGCDigiProducer(edm::ParameterSet const& pset,
                  BunchSpace const& bunchSpace,
                  edm::ProducesCollector,
                  edm::ConsumesCollector& iC);

  void initializeEvent(edm::Event const&, edm::EventSetup const&) override;
  void finalizeEvent(edm::Event&, edm::EventSetup const&) override;
  void accumulate(edm::Event const&, edm::EventSetup const&) override;
  void accumulate(PileUpEventPrincipal const&, edm::EventSetup const&, edm::StreamID const&) override;
  ~HGCDigiProducer() override = default;

private:
  //the digitizer
  bool premixStage1_, premixStage2_;
  HGCDigitizer theDigitizer_;
  CLHEP::HepRandomEngine* randomEngine_ = nullptr;
};

#endif
