#ifndef HitQuadrupletGeneratorFromTripletAndLayers_H
#define HitQuadrupletGeneratorFromTripletAndLayers_H

/** A HitQuadrupletGenerator from HitTripletGenerator and vector of
    Layers. The HitTripletGenerator provides a set of hit triplets.
    For each triplet the search for compatible hit(s) is done among
    provided Layers
 */

#include "RecoPixelVertexing/PixelTriplets/interface/HitQuadrupletGenerator.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGenerator.h"
#include <vector>
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"

#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"


class HitQuadrupletGeneratorFromTripletAndLayers : public HitQuadrupletGenerator {

public:
  typedef LayerHitMapCache  LayerCacheType;

  virtual ~HitQuadrupletGeneratorFromTripletAndLayers() {}
  virtual void init( std::unique_ptr<HitTripletGenerator> triplets, 
    const std::vector<ctfseeding::SeedingLayer>& layers, LayerCacheType* layerCache) = 0; 
};
#endif


