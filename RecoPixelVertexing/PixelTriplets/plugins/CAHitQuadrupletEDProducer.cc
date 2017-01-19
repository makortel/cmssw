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
#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitTriplets.h"
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
  edm::RunningAverage localRATriplets_;

  CAHitQuadrupletGenerator generator_;
};

CAHitQuadrupletEDProducer::CAHitQuadrupletEDProducer(const edm::ParameterSet& iConfig):
  doubletToken_(consumes<IntermediateHitDoublets>(iConfig.getParameter<edm::InputTag>("doublets"))),
  generator_(iConfig, consumesCollector(), false)
{
  produces<RegionsSeedingHitSets>();
  if(generator_.produceTriplets()) {
    produces<RegionsSeedingHitSets>("triplets");
  }
}

void CAHitQuadrupletEDProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("doublets", edm::InputTag("hitPairEDProducer"));
  CAHitQuadrupletGenerator::fillDescriptions(desc);

  auto label = CAHitQuadrupletGenerator::fillDescriptionsLabel() + std::string("EDProducer");
  descriptions.add(label, desc);
}

void CAHitQuadrupletEDProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<IntermediateHitDoublets> hdoublets;
  iEvent.getByToken(doubletToken_, hdoublets);
  const auto& regionDoublets = *hdoublets;

  const SeedingLayerSetsHits& seedingLayerHits = regionDoublets.seedingLayerHits();
  if(seedingLayerHits.numberOfLayersInSet() < generator_.minLayers) {
    throw cms::Exception("LogicError") << "CAHitQuadrupletEDProducer expects SeedingLayerSetsHits::numberOfLayersInSet() to be >= " << generator_.minLayers << ", got " << seedingLayerHits.numberOfLayersInSet() << ". This is likely caused by a configuration error of this module, HitPairEDProducer, or SeedingLayersEDProducer.";
  }

  auto seedingHitSets = std::make_unique<RegionsSeedingHitSets>();
  auto seedingHitSetsTriplets = std::make_unique<RegionsSeedingHitSets>(); // for triplets when producing both triplets and quadruplets
  if(regionDoublets.empty()) {
    iEvent.put(std::move(seedingHitSets));
    if(generator_.produceTriplets()) {
      iEvent.put(std::move(seedingHitSetsTriplets), "triplets");
    }
    return;
  }
  seedingHitSets->reserve(regionDoublets.regionSize(), localRA_.upper());
  if(generator_.produceTriplets()) {
    seedingHitSetsTriplets->reserve(regionDoublets.regionSize(), localRATriplets_.upper());
  }
  generator_.initEvent(iEvent, iSetup);

  LogDebug("CAHitQuadrupletEDProducer") << "Creating ntuplets for " << regionDoublets.regionSize() << " regions, and " << regionDoublets.layerPairsSize() << " layer pairs";

  OrderedHitSeeds quadruplets, triplets;
  quadruplets.reserve(localRA_.upper());
  if(generator_.produceTriplets()) {
    triplets.reserve(localRATriplets_.upper());
  }

  for(const auto& regionLayerPairs: regionDoublets) {
    const TrackingRegion& region = regionLayerPairs.region();
    auto seedingHitSetsFiller = seedingHitSets->beginRegion(&region);
    auto seedingHitSetsTripletsFiller = seedingHitSetsTriplets->dummyFiller();
    if(generator_.produceTriplets()) {
      seedingHitSetsTripletsFiller = seedingHitSetsTriplets->beginRegion(&region);
    }

    LogTrace("CAHitQuadrupletEDProducer") << " starting region";

    generator_.hitNtuplets(regionLayerPairs, quadruplets, triplets, iSetup, seedingLayerHits);
    LogTrace("CAHitQuadrupletEDProducer") << "  created " << quadruplets.size() << " quadrupets and " << triplets.size() << " triplets";
    for(const auto& quad: quadruplets) {
      seedingHitSetsFiller.emplace_back(quad[0], quad[1], quad[2], quad[3]);
    }
    if(generator_.produceTriplets()) {
      for(const auto& triplet: triplets) {
        seedingHitSetsTripletsFiller.emplace_back(triplet[0], triplet[1], triplet[2]);
      }
    }

    quadruplets.clear();
    triplets.clear();
  }
  localRA_.update(seedingHitSets->size());
  localRATriplets_.update(seedingHitSetsTriplets->size());

  iEvent.put(std::move(seedingHitSets));
  if(generator_.produceTriplets()) {
    iEvent.put(std::move(seedingHitSetsTriplets), "triplets");
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CAHitQuadrupletEDProducer);
