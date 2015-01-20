#include "LayerQuadruplets.h"

using namespace ctfseeding;
std::vector<LayerQuadruplets::LayerTripletAndLayers> LayerQuadruplets::layers() const
{
  std::vector<LayerTripletAndLayers> result;

  for(const SeedingLayers& set: theSets) {
    if (set.size() != 4) continue;
    SeedingLayerTriplet layerTriplet = std::make_tuple(set[0], set[1], set[2]);
    bool added = false;
    for(LayerTripletAndLayers& ir: result) {
      const SeedingLayerTriplet & resTriplet = std::get<0>(ir);
      if (std::get<0>(resTriplet) == std::get<0>(layerTriplet) &&
          std::get<1>(resTriplet) == std::get<1>(layerTriplet) &&
          std::get<2>(resTriplet) == std::get<2>(layerTriplet)) {
        std::vector<SeedingLayer>& fourths = std::get<1>(ir);
        fourths.push_back( set[3] );
        added = true;
        break;
      }
    }
    if (!added) {
      LayerTripletAndLayers ltl = std::make_tuple(layerTriplet,  std::vector<SeedingLayer>(1, set[3]) );
      result.push_back(ltl);
    }
  }
  return result;
}
