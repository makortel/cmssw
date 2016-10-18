#ifndef RecoTracker_TkTrackingRegions_TrackingRegionEDProducerT_H
#define RecoTracker_TkTrackingRegions_TrackingRegionEDProducerT_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionFwd.h"
#include "DataFormats/Common/interface/OwnVector.h"

template <typename T_TrackingRegionProducer>
class TrackingRegionEDProducerT: public edm::stream::EDProducer<> {
public:
  using ProductType = TrackingRegionCollection;

  TrackingRegionEDProducerT(const edm::ParameterSet& iConfig):
    regionProducer_(iConfig, consumesCollector()) {
    produces<ProductType>();
  }

  ~TrackingRegionEDProducerT() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    T_TrackingRegionProducer::fillDescriptions(descriptions);
  }

  virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override {
    auto regions = regionProducer_.regions(iEvent, iSetup);
    auto ret = std::make_unique<ProductType>();
    ret->reserve(regions.size());
    for(auto& regionPtr: regions) {
      ret->emplace_back(regionPtr.release());
    }

    iEvent.put(std::move(ret));
  }

private:
  T_TrackingRegionProducer regionProducer_;
};

#endif
