#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitorFactory.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"

#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"

template <typename T_SeedCreator>
class SeedGeneratorFromSeedingHitsEDProducerT: public edm::stream::EDProducer<> {
public:

  SeedGeneratorFromSeedingHitsEDProducerT(const edm::ParameterSet& iConfig);
  ~SeedGeneratorFromSeedingHitsEDProducerT();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  edm::EDGetTokenT<std::vector<SeedingHitSet> > seedingHitSetsToken_;
  SeedCreator seedCreator_;
};

template <typename T_SeedCreator>
SeedGeneratorFromSeedingHitsEDProducerT<T_SeedCreator>::SeedGeneratorFromSeedingHitsEDProducerT(const edm::ParameterSet& iConfig):
  seedingHitSetsToken_(consumes<std::vector<SeedingHitSet> >(iConfig.getParameter<edm::InputTag>("seedingHitSets"))),
  seedCreator_(iConfig)
{
  produces<TrajectorySeedCollection>();
}

template <typename T_SeedCreator>
void SeedGeneratorFromSeedingHitsEDProducerT<T_SeedCreator>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("seedingHitSets", edm::InputTag("hitPairEDProducer"));

  T_SeedCreator::fillDescriptions(desc);

  descriptions.add(T_SeedCreator::fillDescriptionsLabel(), desc);
}


template <typename T_SeedCreator>
SeedGeneratorFromSeedingHitsEDProducerT<T_SeedCreator>::~SeedGeneratorFromSeedingHitsEDProducerT() {}

template <typename T_SeedCreator>
void SeedGeneratorFromSeedingHitsEDProducerT<T_SeedCreator>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<std::vector<SeedingHitSet> > hseedingHitSets;
  iEvent.getByToken(seedingHitSetsToken_, hseedingHitSets);
  const auto& seedingHitSets = *hseedingHitSets;


  auto seeds = std::make_unique<TrajectorySeedCollection>();
  seeds->reserve(seedingHitSets->size());

  

  seeds->shrink_to_fit();
  iEvent.put(std::move(seeds));
}
