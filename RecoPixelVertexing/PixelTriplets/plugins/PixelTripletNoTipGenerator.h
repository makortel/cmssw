#ifndef PixelTripletNoTipGenerator_H
#define PixelTripletNoTipGenerator_H

#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "CombinedHitTripletGenerator.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"

namespace edm { class Event; class EventSetup; } 

#include <utility>
#include <vector>


class PixelTripletNoTipGenerator : public HitTripletGeneratorFromPairAndLayers {
typedef CombinedHitTripletGenerator::LayerCacheType       LayerCacheType;
public:
  PixelTripletNoTipGenerator(const edm::ParameterSet& cfg);

  virtual ~PixelTripletNoTipGenerator() { delete thePairGenerator; }

  void setSeedingLayers(SeedingLayerSetNew::SeedingLayers pairLayers,
                        std::vector<SeedingLayerSetNew::SeedingLayer> thirdLayers) override;

  void init( const HitPairGenerator & pairs, LayerCacheType* layerCache) override;

  virtual void hitTriplets( const TrackingRegion& region, OrderedHitTriplets & trs,
      const edm::Event & ev, const edm::EventSetup& es);

  const HitPairGenerator & pairGenerator() const { return *thePairGenerator; }

private:
  HitPairGenerator * thePairGenerator;
  std::vector<SeedingLayerSetNew::SeedingLayer> theLayers;
  LayerCacheType * theLayerCache;
  float extraHitRZtolerance;
  float extraHitRPhitolerance;
  float extraHitPhiToleranceForPreFiltering;
  double theNSigma;
  double theChi2Cut;
};
#endif
