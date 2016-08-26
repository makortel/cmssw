#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/RunningAverage.h"

#include "RecoTracker/TkHitPairs/interface/RegionsSeedingHitSets.h"
#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitSeeds.h"
#include "RecoTracker/TkHitPairs/interface/IntermediateHitDoublets.h"

#include "CAHitQuadrupletGenerator.h"

class CAHitQuadrupletEDProducer: public edm::stream::EDProducer<> {
public:
  CAHitQuadrupletEDProducer(const edm::ParameterSet& iConfig);
  ~CAHitQuadrupletEDProducer() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  edm::EDGetTokenT<IntermediateHitDoublets> doubletToken_;

  edm::RunningAverage localRA_;

  CAHitQuadrupletGenerator generator_;
};

CAHitQuadrupletEDProducer::CAHitQuadrupletEDProducer(const edm::ParameterSet& iConfig):
  doubletToken_(consumes<IntermediateHitDoublets>(iConfig.getParameter<edm::InputTag>("doublets"))),
  generator_(iConfig, consumesCollector(), false)
{
  produces<RegionsSeedingHitSets>();
}

void CAHitQuadrupletEDProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("doublets", edm::InputTag("hitPairEDProducer"));
  CAHitQuadrupletGenerator::fillDescriptions(desc);

  descriptions.add("caHitQuadrupletEDProducer", desc);
}

void CAHitQuadrupletEDProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<IntermediateHitDoublets> hdoublets;
  iEvent.getByToken(doubletToken_, hdoublets);
  const auto& regionDoublets = *hdoublets;

  const SeedingLayerSetsHits& seedingLayerHits = regionDoublets.seedingLayerHits();
  if(seedingLayerHits.numberOfLayersInSet() < 4) {
    throw cms::Exception("Configuration") << "CAHitQuadrupletEDProducer expects SeedingLayerSetsHits::numberOfLayersInSet() to be >= 4, got " << seedingLayerHits.numberOfLayersInSet();
  }

  auto seedingHitSets = std::make_unique<RegionsSeedingHitSets>();
  if(regionDoublets.empty()) {
    iEvent.put(std::move(seedingHitSets));
    return;
  }
  seedingHitSets->reserve(regionDoublets.regionSize(), localRA_.upper());
  generator_.initEvent(iEvent, iSetup);

  LogDebug("CAHitQuadrupletEDProducer") << "Creating quadruplets for " << regionDoublets.regionSize() << " regions, and " << regionDoublets.layerPairsSize() << " layer pairs";

  OrderedHitSeeds quadruplets;
  quadruplets.reserve(localRA_.upper());

  std::vector<const HitDoublets *> layersDoublets;
  layersDoublets.reserve(3);

  auto layerPairEqual = [](const IntermediateHitDoublets::LayerPairHitDoublets& pair,
                           SeedingLayerSetsHits::LayerIndex inner,
                           SeedingLayerSetsHits::LayerIndex outer) {
    return pair.innerLayerIndex() == inner && pair.outerLayerIndex() == outer;
  };

  for(const auto& regionLayerPairs: regionDoublets) {
    const TrackingRegion& region = regionLayerPairs.region();
    auto seedingHitSetsFiller = seedingHitSets->beginRegion(&region);

    LogTrace("CAHitQuadrupletEDProducer") << " starting region";

    // Probably there is a better way to organize the delivery of doublets?
    for(const auto& layerQuad: seedingLayerHits) {
      LogTrace("CAHitQuadrupletEDProducer") << "  starting layer quadruplet "
                                            << layerQuad[0].index() << "," << layerQuad[1].index() << "," << layerQuad[2].index() << "," << layerQuad[3].index();

      using namespace std::placeholders;
      auto found1 = std::find_if(regionLayerPairs.begin(), regionLayerPairs.end(), std::bind(layerPairEqual, _1, layerQuad[0].index(), layerQuad[1].index()));
      if(found1 == regionLayerPairs.end()) // no hits from this layer pair
        continue;
      auto found2 = std::find_if(regionLayerPairs.begin(), regionLayerPairs.end(), std::bind(layerPairEqual, _1, layerQuad[1].index(), layerQuad[2].index()));
      if(found2 == regionLayerPairs.end()) // no hits from this layer pair
        continue;
      auto found3 = std::find_if(regionLayerPairs.begin(), regionLayerPairs.end(), std::bind(layerPairEqual, _1, layerQuad[2].index(), layerQuad[3].index()));
      if(found3 == regionLayerPairs.end()) // no hits from this layer pair
        continue;

      layersDoublets.push_back(&(found1->doublets()));
      layersDoublets.push_back(&(found2->doublets()));
      layersDoublets.push_back(&(found3->doublets()));

      generator_.hitQuadruplets(region, quadruplets, iSetup, layersDoublets, layerQuad);

      LogTrace("CAHitQuadrupletEDProducer") << "  created " << quadruplets.size() << " quadruplets";

      for(const auto& quad: quadruplets) {
        seedingHitSetsFiller.emplace_back(quad[0], quad[1], quad[2], quad[3]);
      }
      quadruplets.clear();
      layersDoublets.clear();
    }
  }
  localRA_.update(seedingHitSets->size());

  iEvent.put(std::move(seedingHitSets));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CAHitQuadrupletEDProducer);
