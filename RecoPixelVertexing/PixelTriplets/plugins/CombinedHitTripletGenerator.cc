#include "CombinedHitTripletGenerator.h"

#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayersFactory.h"
#include "RecoPixelVertexing/PixelTriplets/interface/LayerTriplets.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

CombinedHitTripletGenerator::CombinedHitTripletGenerator(const edm::ParameterSet& cfg) :
  theSeedingLayerSrc(cfg.getParameter<edm::InputTag>("SeedingLayers"))
{
  edm::ParameterSet generatorPSet = cfg.getParameter<edm::ParameterSet>("GeneratorPSet");
  std::string       generatorName = generatorPSet.getParameter<std::string>("ComponentName");
  theGenerator.reset(HitTripletGeneratorFromPairAndLayersFactory::get()->create(generatorName, generatorPSet));
  theGenerator->init(HitPairGeneratorFromLayerPair(0, 1, &theLayerCache), &theLayerCache);
}

CombinedHitTripletGenerator::~CombinedHitTripletGenerator() {}

void CombinedHitTripletGenerator::hitTriplets(
   const TrackingRegion& region, OrderedHitTriplets & result,
   const edm::Event& ev, const edm::EventSetup& es)
{
  edm::Handle<SeedingLayerSetsHits> hlayers;
  ev.getByLabel(theSeedingLayerSrc, hlayers);
  assert(hlayers->numberOfLayersInSet() == 3);

  std::vector<LayerTriplets::LayerSetAndLayers> trilayers = LayerTriplets::layers(*hlayers);
  for(const auto& setAndLayers: trilayers) {
    theGenerator->setSeedingLayers(setAndLayers.first, setAndLayers.second);
    theGenerator->hitTriplets( region, result, ev, es);
  }
  theLayerCache.clear();
}

