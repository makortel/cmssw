#include "CombinedMultiHitGenerator.h"

#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "RecoTracker/TkSeedGenerator/interface/MultiHitGeneratorFromPairAndLayers.h"
#include "RecoTracker/TkSeedGenerator/interface/MultiHitGeneratorFromPairAndLayersFactory.h"
#include "RecoPixelVertexing/PixelTriplets/interface/LayerTriplets.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

CombinedMultiHitGenerator::CombinedMultiHitGenerator(const edm::ParameterSet& cfg):
  theSeedingLayerSrc(cfg.getParameter<edm::InputTag>("SeedingLayers"))
{
  edm::ParameterSet generatorPSet = cfg.getParameter<edm::ParameterSet>("GeneratorPSet");
  std::string       generatorName = generatorPSet.getParameter<std::string>("ComponentName");
  theGenerator.reset(MultiHitGeneratorFromPairAndLayersFactory::get()->create(generatorName, generatorPSet));
  theGenerator->init(HitPairGeneratorFromLayerPair( 0, 1, &theLayerCache), &theLayerCache);
}

CombinedMultiHitGenerator::~CombinedMultiHitGenerator() {}

void CombinedMultiHitGenerator::hitSets(
   const TrackingRegion& region, OrderedMultiHits & result,
   const edm::Event& ev, const edm::EventSetup& es)
{
  edm::Handle<SeedingLayerSetsHits> hlayers;
  ev.getByLabel(theSeedingLayerSrc, hlayers);
  assert(hlayers->numberOfLayersInSet() == 3);

  std::vector<LayerTriplets::LayerSetAndLayers> trilayers = LayerTriplets::layers(*hlayers);
  for(const auto& setAndLayers: trilayers) {
    theGenerator->setSeedingLayers(setAndLayers.first, setAndLayers.second);
    theGenerator->hitSets( region, result, ev, es);
  }
  theLayerCache.clear();
}

