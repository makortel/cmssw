#include "RecoTracker/TkHitPairs/interface/CombinedHitPairGenerator.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

CombinedHitPairGenerator::CombinedHitPairGenerator(const edm::ParameterSet& cfg)
  : theSeedingLayerSrc(cfg.getParameter<edm::InputTag>("SeedingLayers"))
{
  theMaxElement = cfg.getParameter<unsigned int>("maxElement");
  theGenerator.reset(new HitPairGeneratorFromLayerPair(0, 1, &theLayerCache, theMaxElement));
}

CombinedHitPairGenerator::CombinedHitPairGenerator(const CombinedHitPairGenerator& cb):
  theSeedingLayerSrc(cb.theSeedingLayerSrc),
  theGenerator(new HitPairGeneratorFromLayerPairNew(0, 1, &theLayerCache, cb.theMaxElement))
{
  theMaxElement = cb.theMaxElement;
}

CombinedHitPairGenerator::~CombinedHitPairGenerator() {}

void CombinedHitPairGenerator::setSeedingLayers(SeedingLayerSetNew::SeedingLayers layers) {
  assert(0 && "not implemented");
}

void CombinedHitPairGenerator::hitPairs(
   const TrackingRegion& region, OrderedHitPairs  & result,
   const edm::Event& ev, const edm::EventSetup& es)
{
  edm::Handle<SeedingLayerSetNew> hlayers;
  ev.getByLabel(theSeedingLayerSrc, hlayers);
  assert(hlayers->sizeLayers() == 2);
  for(unsigned i=0; i<hlayers->sizeLayerSets(); ++i) {
    theGenerator->setSeedingLayers(hlayers->getLayers(i));
    theGenerator->hitPairs( region, result, ev, es);
  }

  theLayerCache.clear();

  LogDebug("CombinedHitPairGenerator")<<" total number of pairs provided back CHPG : "<<result.size();

}
