#include "RecoPixelVertexing/PixelTriplets/interface/LayerTriplets.h"

using namespace ctfseeding;
std::vector<LayerTriplets::LayerPairAndLayers> LayerTriplets::layers() const
{
  std::vector<LayerPairAndLayers> result;
  typedef std::vector<LayerPairAndLayers>::iterator IR;

  typedef SeedingLayerSets::const_iterator IL;
  for (IL il=theSets.begin(), ilEnd= theSets.end(); il != ilEnd; ++il) {
    const SeedingLayers & set = *il;
    if (set.size() != 3) continue;
    SeedingLayerPair layerPair = std::make_pair(set[0], set[1]);
    bool added = false;
    for (IR ir = result.begin(); ir < result.end(); ++ir) {
      const SeedingLayerPair & resPair = ir->first;
      if (resPair.first ==layerPair.first && resPair.second == layerPair.second) {
        std::vector<SeedingLayer> & thirds = ir->second;
        thirds.push_back( set[2] );
        added = true;
        break;
      }
    }
    if (!added) {
      LayerPairAndLayers lpl = std::make_pair(layerPair,  std::vector<SeedingLayer>(1, set[2]) );
      result.push_back(lpl);
    }
  }
  return result;
}

namespace layerTripletsNew {
  std::vector<LayerSetAndLayers> layers(const SeedingLayerSetNew& sets) {
    std::vector<LayerSetAndLayers> result;
    if(sets.sizeLayers() != 3)
      return result;

    for(unsigned int iLayers=0; iLayers < sets.sizeLayerSets(); ++iLayers) {
      LayerSet set = sets.getLayers(iLayers);
      bool added = false;

      for(auto ir = result.begin(); ir < result.end(); ++ir) {
        const LayerSet & resSet = ir->first;
        if (resSet.getLayer(0).index() == set.getLayer(0).index() && resSet.getLayer(1).index() == set.getLayer(1).index()) {
          std::vector<Layer>& thirds = ir->second;
          thirds.push_back( set.getLayer(2) );
          added = true;
          break;
        }
      }
      if (!added) {
        LayerSetAndLayers lpl = std::make_pair(set,  std::vector<Layer>(1, set.getLayer(2)) );
        result.push_back(lpl);
      }
    }
    return result;
  }
}
