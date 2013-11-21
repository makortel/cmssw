#include "CombinedHitTripletGenerator.h"

#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSets.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSetsBuilder.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayersFactory.h"
#include "RecoPixelVertexing/PixelTriplets/interface/LayerTriplets.h"

#include <string>

using namespace std;
using namespace ctfseeding;

CombinedHitTripletGenerator::CombinedHitTripletGenerator(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC)
{
  edm::ParameterSet generatorPSet = cfg.getParameter<edm::ParameterSet>("GeneratorPSet");
  std::string       generatorName = generatorPSet.getParameter<std::string>("ComponentName");
  SeedingLayerSetsBuilder layerBuilder(cfg.getParameter<edm::ParameterSet>("SeedingLayers"), iC);

  SeedingLayerSets layerSets  =  layerBuilder.layers();

  vector<LayerTriplets::LayerPairAndLayers>::const_iterator it;
  vector<LayerTriplets::LayerPairAndLayers> trilayers=LayerTriplets(layerSets).layers();

  for (it = trilayers.begin(); it != trilayers.end(); it++) {
    const SeedingLayer& first = (*it).first.first;
    const SeedingLayer& second = (*it).first.second;
    const vector<SeedingLayer>& thirds = (*it).second;

    std::unique_ptr<HitTripletGeneratorFromPairAndLayers> aGen(HitTripletGeneratorFromPairAndLayersFactory::get()->create(generatorName,generatorPSet, iC));

    aGen->init( HitPairGeneratorFromLayerPair( first, second, &theLayerCache),
                thirds, &theLayerCache);

    theGenerators.push_back(std::move(aGen));
  }
}

CombinedHitTripletGenerator::~CombinedHitTripletGenerator() {}


void CombinedHitTripletGenerator::hitTriplets(
   const TrackingRegion& region, OrderedHitTriplets & result,
   const edm::Event& ev, const edm::EventSetup& es)
{
  GeneratorContainer::const_iterator i;
  for (i=theGenerators.begin(); i!=theGenerators.end(); i++) {
    (**i).hitTriplets( region, result, ev, es);
  }
  theLayerCache.clear();
}

