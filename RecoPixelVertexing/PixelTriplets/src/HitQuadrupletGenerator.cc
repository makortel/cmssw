#include "RecoPixelVertexing/PixelTriplets/interface/HitQuadrupletGenerator.h"

HitQuadrupletGenerator::HitQuadrupletGenerator(unsigned int nSize)
{
  theQuadruplets.reserve(nSize);
}

const OrderedHitSeeds & HitQuadrupletGenerator::run(
    const TrackingRegion& region, const edm::Event & ev, const edm::EventSetup& es)
{
  theQuadruplets.clear();
  hitQuadruplets(region, theQuadruplets, ev, es);
  return theQuadruplets;
}

void HitQuadrupletGenerator::clear() 
{
  theQuadruplets.clear();
}

