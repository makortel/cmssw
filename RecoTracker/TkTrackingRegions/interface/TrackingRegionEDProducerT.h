#ifndef RecoTracker_TkTrackingRegions_TrackingRegionEDProducerT_H
#define RecoTracker_TkTrackingRegions_TrackingRegionEDProducerT_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"

template <typename T_TrackingRegionProducer>
class TrackingRegionEDProducerT: public edm::stream::EDProducer<> {
public:
  using ProductType = std::vector<std::unique_ptr<TrackingRegion>>;

  TrackingRegionEDProducerT(const edm::ParameterSet& iConfig):
    regionProducer_(iConfig, consumesCollector()) {
    produces<ProductType>();
  }

  ~TrackingRegionEDProducerT() = default;

  virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override {
    auto regions = std::make_unique<ProductType>(regionProducer_.regions(iEvent, iSetup));
    iEvent.put(std::move(regions));
  }

private:
  T_TrackingRegionProducer regionProducer_;
};

#endif
