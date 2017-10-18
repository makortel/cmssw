#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/RunningAverage.h"

#include "RecoTracker/TkHitPairs/interface/RegionsSeedingHitSets.h"
#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitSeeds.h"
#include "RecoTracker/TkHitPairs/interface/IntermediateHitDoublets.h"

#include "CAHitNtupletGenerator.h"


namespace {
  void fillNtuplets(RegionsSeedingHitSets::RegionFiller& seedingHitSetsFiller,
                    const OrderedHitSeeds& quadruplets) {
    for(const auto& quad: quadruplets) {
      seedingHitSetsFiller.emplace_back(quad[0], quad[1], quad[2], quad[3]);
    }
  }
}

class CAHitNtupletEDProducer: public edm::stream::EDProducer<> {
public:
  CAHitNtupletEDProducer(const edm::ParameterSet& iConfig);
  ~CAHitNtupletEDProducer() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  edm::EDGetTokenT<IntermediateHitDoublets> doubletToken_;
  std::vector<unsigned int> forNLayers_;

  edm::RunningAverage localRA_;

  CAHitNtupletGenerator generator_;
};

CAHitNtupletEDProducer::CAHitNtupletEDProducer(const edm::ParameterSet& iConfig):
  doubletToken_(consumes<IntermediateHitDoublets>(iConfig.getParameter<edm::InputTag>("doublets"))),
  forNLayers_(iConfig.getParameter<std::vector<unsigned int> >("forNLayers")),
  generator_(iConfig, consumesCollector()) // TODO: how to deliver the "number of layers" to the generator?
{
  // Sanity checks
  if(forNLayers_.empty()) throw cms::Exception("Configuration") << "forNLayers is empty, need at least one element";
  unsigned prev = forNLayers_[0];
  for(size_t i=1, end=forNLayers_.size(); i<end; ++i) {
    if(forNLayers_[i] <= prev) throw cms::Exception("Configuration") << "Elements of forNLayers must be in ascending order, now element " << i << " is " << forNLayers_[i] << " while the element " << (i-1) << " was " << prev;
    prev = forNLayers_[i];
  }

  // Setup output
  for(auto nlayers: forNLayers_) {
    produces<RegionsSeedingHitSets>(std::to_string(nlayers));
  }
}

void CAHitNtupletEDProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("doublets", edm::InputTag("hitPairEDProducer"));
  desc.add<std::vector<unsigned int>>("forNLayers", std::vector<unsigned int>{{4}})->setComment("Produce separate collections of ntuplets for each \"number of layers\", i.e. triplets for 3, quadruplets for 4 etc. The last number is interpreted as \"ntuplets from at least this many layers\"."); // by default quadruplets only
  CAHitNtupletGenerator::fillDescriptions(desc);

  auto label = CAHitNtupletGenerator::fillDescriptionsLabel() + std::string("EDProducer");
  descriptions.add(label, desc);
}

void CAHitNtupletEDProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<IntermediateHitDoublets> hdoublets;
  iEvent.getByToken(doubletToken_, hdoublets);
  const auto& regionDoublets = *hdoublets;

  const SeedingLayerSetsHits& seedingLayerHits = regionDoublets.seedingLayerHits();
  if(seedingLayerHits.numberOfLayersInSet() < CAHitNtupletGenerator::minLayers) {
    throw cms::Exception("LogicError") << "CAHitNtupletEDProducer expects SeedingLayerSetsHits::numberOfLayersInSet() to be >= " << CAHitNtupletGenerator::minLayers << ", got " << seedingLayerHits.numberOfLayersInSet() << ". This is likely caused by a configuration error of this module, HitPairEDProducer, or SeedingLayersEDProducer.";
  }

  std::vector<std::unique_ptr<RegionsSeedingHitSets> > seedingHitSets(forNLayers_.size());
  for(size_t i=0, end=forNLayers_.size(); i<end; ++i) {
    seedingHitSets[i] = std::make_unique<RegionsSeedingHitSets>();
  }
  if(regionDoublets.empty()) {
    for(size_t i=0, end=forNLayers_.size(); i<end; ++i) {
      iEvent.put(std::move(seedingHitSets[i]), std::to_string(forNLayers_[i]));
    }
    return;
  }
  for(size_t i=0, end=forNLayers_.size(); i<end; ++i) {
    seedingHitSets[i]->reserve(regionDoublets.regionSize(), localRA_.upper());
  }
  generator_.initEvent(iEvent, iSetup);

  LogDebug("CAHitNtupletEDProducer") << "Creating ntuplets for " << regionDoublets.regionSize() << " regions, and " << regionDoublets.layerPairsSize() << " layer pairs";
  std::vector<OrderedHitSeeds> ntuplets;
  ntuplets.resize(regionDoublets.regionSize());
  for(auto& ntuplet : ntuplets)  ntuplet.reserve(localRA_.upper());

  generator_.hitNtuplets(regionDoublets, ntuplets, iSetup, seedingLayerHits); // TODO: how exactly to communicate with the generator about the different layer-number ntuplets?
  int index = 0;
  for(const auto& regionLayerPairs: regionDoublets) {
    const TrackingRegion& region = regionLayerPairs.region();
    auto seedingHitSetsFiller = seedingHitSets[0]->beginRegion(&region); // TODO: eventually must have a loop over seedingHitSets

    fillNtuplets(seedingHitSetsFiller, ntuplets[index]); // TODO: need to generalize this as well
    ntuplets[index].clear();
    index++;
  }

  size_t maxSize=0;
  for(size_t i=0, end=forNLayers_.size(); i<end; ++i) {
    maxSize = std::max(maxSize, seedingHitSets[i]->size());
    iEvent.put(std::move(seedingHitSets[i]), std::to_string(forNLayers_[i]));
  }
  localRA_.update(maxSize);
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CAHitNtupletEDProducer);
