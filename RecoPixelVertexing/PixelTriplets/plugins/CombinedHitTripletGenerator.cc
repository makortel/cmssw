#include "CombinedHitTripletGenerator.h"

#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSets.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSetsBuilder.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayersFactory.h"
#include "RecoPixelVertexing/PixelTriplets/interface/LayerTriplets.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"


using namespace std;
using namespace ctfseeding;

CombinedHitTripletGenerator::CombinedHitTripletGenerator(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC)
  : initialised(false),
    theLayerBuilderName(cfg.getParameter<std::string>("SeedingLayers")),
    theLayerBuilder(cfg.getParameter<edm::ParameterSet>("SeedingLayers"))
{
  edm::ParameterSet generatorPSet = cfg.getParameter<edm::ParameterSet>("GeneratorPSet");
  std::string       generatorName = generatorPSet.getParameter<std::string>("ComponentName");
  theGeneratorPrototype.reset(HitTripletGeneratorFromPairAndLayersFactory::get()->create(generatorName,generatorPSet, iC));
}

void CombinedHitTripletGenerator::init(const edm::EventSetup& es)
{
//  edm::ParameterSet leyerPSet = cfg.getParameter<edm::ParameterSet>("LayerPSet");
//  SeedingLayerSets layerSets  = SeedingLayerSetsBuilder(leyerPSet).layers(es);

  //edm::ESHandle<SeedingLayerSetsBuilder> layerBuilder;
  //es.get<TrackerDigiGeometryRecord>().get(theLayerBuilderName, layerBuilder);

  SeedingLayerSets layerSets  =  theLayerBuilder.layers(es);


  vector<LayerTriplets::LayerPairAndLayers>::const_iterator it;
  vector<LayerTriplets::LayerPairAndLayers> trilayers=LayerTriplets(layerSets).layers();

  for (it = trilayers.begin(); it != trilayers.end(); it++) {
    SeedingLayer first = (*it).first.first;
    SeedingLayer second = (*it).first.second;
    vector<SeedingLayer> thirds = (*it).second;


    std::unique_ptr<HitTripletGeneratorFromPairAndLayers> aGen(theGeneratorPrototype->clone());

    aGen->init( HitPairGeneratorFromLayerPair( first, second, &theLayerCache),
                thirds, &theLayerCache);

    theGenerators.push_back(std::move(aGen));
  }

  initialised = true;

}

CombinedHitTripletGenerator::~CombinedHitTripletGenerator() {}


void CombinedHitTripletGenerator::hitTriplets(
   const TrackingRegion& region, OrderedHitTriplets & result,
   const edm::Event& ev, const edm::EventSetup& es)
{
  if (!initialised) init(es);

  GeneratorContainer::const_iterator i;
  for (i=theGenerators.begin(); i!=theGenerators.end(); i++) {
    (**i).hitTriplets( region, result, ev, es);
  }
  theLayerCache.clear();
}

