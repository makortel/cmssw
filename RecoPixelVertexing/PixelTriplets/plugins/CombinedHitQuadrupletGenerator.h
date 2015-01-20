#ifndef CombinedHitQuadrupletGenerator_H
#define CombinedHitQuadrupletGenerator_H

/** A HitQuadrupletGenerator consisting of a set of 
 *  triplet generators of type HitQuadrupletGeneratorFromPairAndLayers
 *  initialised from provided layers in the form of PixelLayerQuadruplets  
 */ 

#include <vector>
#include "RecoPixelVertexing/PixelTriplets/interface/HitQuadrupletGenerator.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class TrackingRegion;
class HitQuadrupletGeneratorFromTripletAndLayers;
namespace ctfseeding { class SeedingLayer;}

namespace edm { class Event; }
namespace edm { class EventSetup; }

class CombinedHitQuadrupletGenerator : public HitQuadrupletGenerator {
public:
  typedef LayerHitMapCache  LayerCacheType;

public:

  CombinedHitQuadrupletGenerator( const edm::ParameterSet& cfg);

  virtual ~CombinedHitQuadrupletGenerator();

  /// from base class
  virtual void hitQuadruplets( const TrackingRegion& reg, OrderedHitSeeds & triplets,
      const edm::Event & ev,  const edm::EventSetup& es);

private:
  void init(const edm::ParameterSet & cfg, const edm::EventSetup& es);

  mutable bool initialised;

  edm::ParameterSet         theConfig;
  LayerCacheType            theLayerCache;

  typedef std::vector<std::unique_ptr<HitQuadrupletGeneratorFromTripletAndLayers> > GeneratorContainer;
  GeneratorContainer        theGenerators;
};
#endif
