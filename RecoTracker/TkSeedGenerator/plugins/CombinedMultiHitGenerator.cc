#include "CombinedMultiHitGenerator.h"

#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSets.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSetsBuilder.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "RecoTracker/TkSeedGenerator/interface/MultiHitGeneratorFromPairAndLayers.h"
#include "RecoTracker/TkSeedGenerator/interface/MultiHitGeneratorFromPairAndLayersFactory.h"
#include "RecoPixelVertexing/PixelTriplets/interface/LayerTriplets.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

using namespace std;
using namespace ctfseeding;

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
  edm::Handle<SeedingLayerSetNew> hlayers;
  ev.getByLabel(theSeedingLayerSrc, hlayers);
  assert(hlayers->sizeLayers() == 3);

  std::vector<layerTripletsNew::LayerSetAndLayers> trilayers = layerTripletsNew::layers(*hlayers);
  for(const auto& setAndLayers: trilayers) {
    theGenerator->setSeedingLayers(setAndLayers.first, setAndLayers.second);
    theGenerator->hitSets( region, result, ev, es);
  }
  theLayerCache.clear();
}

