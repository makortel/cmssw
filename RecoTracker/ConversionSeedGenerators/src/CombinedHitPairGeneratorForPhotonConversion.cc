#include "RecoTracker/ConversionSeedGenerators/interface/CombinedHitPairGeneratorForPhotonConversion.h"
#include "RecoTracker/ConversionSeedGenerators/interface/HitPairGeneratorFromLayerPairForPhotonConversion.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

CombinedHitPairGeneratorForPhotonConversion::CombinedHitPairGeneratorForPhotonConversion(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC):
  theSeedingLayerToken(iC.consumes<SeedingLayerSetsHits>(cfg.getParameter<edm::InputTag>("SeedingLayers"))),
  theMaxElement(cfg.getParameter<unsigned int>("maxElement"))
{
  maxHitPairsPerTrackAndGenerator = cfg.getParameter<unsigned int>("maxHitPairsPerTrackAndGenerator");
  theGenerator.reset(new HitPairGeneratorFromLayerPairForPhotonConversion(0, 1, &theLayerCache, maxHitPairsPerTrackAndGenerator));
}

CombinedHitPairGeneratorForPhotonConversion::~CombinedHitPairGeneratorForPhotonConversion() {}

const OrderedHitPairs & CombinedHitPairGeneratorForPhotonConversion::run(
									 const ConversionRegion& convRegion,
									 const TrackingRegion& region, const edm::Event & ev, const edm::EventSetup& es)
{
  thePairs.clear();
  hitPairs(convRegion, region, thePairs, ev, es);
  return thePairs;
}


void CombinedHitPairGeneratorForPhotonConversion::hitPairs(
							   const ConversionRegion& convRegion,
							   const TrackingRegion& region, OrderedHitPairs  & result,
							   const edm::Event& ev, const edm::EventSetup& es)
{
  edm::Handle<SeedingLayerSetsHits> hlayers;
  ev.getByToken(theSeedingLayerToken, hlayers);
  const SeedingLayerSetsHits& layers = *hlayers;
  assert(layers.numberOfLayersInSet() == 2);

  OrderedHitPairs  resultTmp; // why is this needed?
  resultTmp.reserve(maxHitPairsPerTrackAndGenerator);
  for(SeedingLayerSetsHits::LayerSetIndex i=0; i<hlayers->size(); ++i) {
    resultTmp.clear();
    theGenerator->hitPairs( convRegion, region, resultTmp, layers[i], ev, es); // why resultTmp and not result?
    result.insert(result.end(),resultTmp.begin(),resultTmp.end());
  }

  //theLayerCache.clear(); //Don't want to clear now, because have to loop on all the tracks. will be cleared later, calling a specific method
}
