#include "RecoPixelVertexing/PixelTriplets/interface/LayerTriplets.h"

namespace LayerTriplets {
std::vector<LayerSetAndLayers> layers(const SeedingLayerSetsHits& sets) {
  std::vector<LayerSetAndLayers> result;
  if(sets.numberOfLayersInSet() != 3)
    return result;

  for(SeedingLayerSetsHits::LayerSetIndex iLayers=0; iLayers < sets.size(); ++iLayers) {
    LayerSet set = sets[iLayers];
    bool added = false;

    for(auto ir = result.begin(); ir < result.end(); ++ir) {
      const LayerSet & resSet = ir->first;
      if (resSet[0].index() == set[0].index() && resSet[1].index() == set[1].index()) {
        std::vector<Layer>& thirds = ir->second;
        thirds.push_back( set[2] );
        added = true;
        break;
      }
    }
    if (!added) {
      LayerSetAndLayers lpl = std::make_pair(set,  std::vector<Layer>(1, set[2]) );
      result.push_back(lpl);
    }
  }
  return result;
}
}
