#include "CombinedHitQuadrupletGenerator.h"

#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSets.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSetsBuilder.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGenerator.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitQuadrupletGeneratorFromTripletAndLayers.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitQuadrupletGeneratorFromTripletAndLayersFactory.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayersFactory.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "LayerQuadruplets.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"


using namespace std;
using namespace ctfseeding;

CombinedHitQuadrupletGenerator::CombinedHitQuadrupletGenerator(const edm::ParameterSet& cfg)
  : initialised(false), theConfig(cfg)
{ }

void CombinedHitQuadrupletGenerator::init(const edm::ParameterSet & cfg, const edm::EventSetup& es)
{
  std::string layerBuilderName = cfg.getParameter<std::string>("SeedingLayers");
  edm::ESHandle<SeedingLayerSetsBuilder> layerBuilder;
  es.get<TrackerDigiGeometryRecord>().get(layerBuilderName, layerBuilder);

  SeedingLayerSets layerSets  =  layerBuilder->layers(es);

  vector<LayerQuadruplets::LayerTripletAndLayers> quadlayers=LayerQuadruplets(layerSets).layers();

  edm::ParameterSet generatorPSet = theConfig.getParameter<edm::ParameterSet>("GeneratorPSet");
  std::string       generatorName = generatorPSet.getParameter<std::string>("ComponentName");
  edm::ParameterSet tripletGeneratorPSet = theConfig.getParameter<edm::ParameterSet>("TripletGeneratorPSet");
  std::string tripletGeneratorName = tripletGeneratorPSet.getParameter<std::string>("ComponentName");

  for(auto& ltl: quadlayers) {
    auto& triplet = std::get<0>(ltl);

    std::unique_ptr<HitQuadrupletGeneratorFromTripletAndLayers> qGen(HitQuadrupletGeneratorFromTripletAndLayersFactory::get()->create(generatorName, generatorPSet));

    std::unique_ptr<HitTripletGeneratorFromPairAndLayers> tGen(HitTripletGeneratorFromPairAndLayersFactory::get()->create(tripletGeneratorName, tripletGeneratorPSet));

    // Some CPU wasted here because same pairs are generated multiple times
    tGen->init( HitPairGeneratorFromLayerPair( std::get<0>(triplet), std::get<1>(triplet), &theLayerCache),
                std::vector<SeedingLayer>{std::get<2>(triplet)}, &theLayerCache);

    qGen->init(std::move(tGen), std::get<1>(ltl), &theLayerCache);

    theGenerators.push_back(std::move(qGen));
  }

  initialised = true;
}

CombinedHitQuadrupletGenerator::~CombinedHitQuadrupletGenerator() {}

void CombinedHitQuadrupletGenerator::hitQuadruplets(
   const TrackingRegion& region, OrderedHitSeeds & result,
   const edm::Event& ev, const edm::EventSetup& es)
{
  if (!initialised) init(theConfig,es);

  GeneratorContainer::const_iterator i;
  for (i=theGenerators.begin(); i!=theGenerators.end(); i++) {
    (**i).hitQuadruplets( region, result, ev, es);
  }
  theLayerCache.clear();
}

