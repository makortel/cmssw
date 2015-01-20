#ifndef LayerQuadruplets_H
#define LayerQuadruplets_H

/** A class grouping pixel layers in triplets and associating a vector
    of layers to each layer pair. The layer triplet is used to generate
    hit triplets and the associated vector of layers to generate
    a fourth hit confirming layer triplet
 */

#include <vector>
#include <tuple>
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSets.h"

class LayerQuadruplets {
public:
  typedef std::tuple<ctfseeding::SeedingLayer,
                      ctfseeding::SeedingLayer,
                      ctfseeding::SeedingLayer> SeedingLayerTriplet;
  typedef std::tuple<SeedingLayerTriplet, std::vector<ctfseeding::SeedingLayer> > LayerTripletAndLayers;

  LayerQuadruplets( const ctfseeding::SeedingLayerSets & sets) : theSets(sets) {}

  std::vector<LayerTripletAndLayers> layers() const;

private:
  ctfseeding::SeedingLayerSets theSets;
};

#endif

