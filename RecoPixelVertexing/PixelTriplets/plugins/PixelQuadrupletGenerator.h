#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGenerator.h"
#include "CombinedHitQuadrupletGenerator.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitQuadrupletGeneratorFromTripletAndLayers.h"

#include <utility>
#include <vector>

class SeedComparitor;

class PixelQuadrupletGenerator : public HitQuadrupletGeneratorFromTripletAndLayers {

typedef CombinedHitQuadrupletGenerator::LayerCacheType       LayerCacheType;

public:
  PixelQuadrupletGenerator( const edm::ParameterSet& cfg); 

  virtual ~PixelQuadrupletGenerator();

  virtual void init( std::unique_ptr<HitTripletGenerator> triplets,
      const std::vector<ctfseeding::SeedingLayer> & layers, LayerCacheType* layerCache);

  virtual void hitQuadruplets( const TrackingRegion& region, OrderedHitSeeds & trs, 
      const edm::Event & ev, const edm::EventSetup& es);

private:
  std::unique_ptr<HitTripletGenerator> theTripletGenerator;
  std::vector<ctfseeding::SeedingLayer> theLayers;
  LayerCacheType * theLayerCache;

  std::unique_ptr<SeedComparitor> theComparitor;

  const double extraHitRZtolerance;
  const double extraHitRPhitolerance;
  const double maxChi2;
  const bool keepTriplets;
};



