/** \class AlignmentMuonSelectorModule
 *
 * selects a subset of a muon collection and clones
 * Track, TrackExtra parts and RecHits collection
 * for SA, GB and Tracker Only options
 *
 * \author Javier Fernandez, IFCA
 *
 * \version $Revision: 1.3 $
 *
 * $Id: AlignmentMuonSelectorModule.cc,v 1.3 2008/02/04 19:32:26 flucke Exp $
 *
 */

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"
#include "Alignment/CommonAlignmentProducer/interface/AlignmentMuonSelector.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

// the following include is necessary to clone all track branches
// including recoTrackExtras and TrackingRecHitsOwned.
// if you remove it the code will compile, but the cloned
// tracks have only the recoMuons branch!

struct MuonConfigSelector {

  typedef std::vector<const reco::Muon*> container;
  typedef container::const_iterator const_iterator;
  typedef reco::MuonCollection collection;

  MuonConfigSelector( const edm::ParameterSet & cfg, edm::ConsumesCollector && iC ) :
    theSelector(cfg) {}

  const_iterator begin() const { return selected_.begin(); }
  const_iterator end() const { return selected_.end(); }
  size_t size() const { return selected_.size(); }

  void select( const edm::Handle<reco::MuonCollection> & c,  const edm::Event & evt, const edm::EventSetup &/* dummy*/)
  {
    all_.clear();
    selected_.clear();
    for (collection::const_iterator i = c.product()->begin(), iE = c.product()->end();
         i != iE; ++i){
      all_.push_back(& * i );
    }
    selected_ = theSelector.select(all_, evt); // might add dummy
  }

private:
  container all_,selected_;
  AlignmentMuonSelector theSelector;
};

typedef ObjectSelector<MuonConfigSelector>  AlignmentMuonSelectorModule;

template<>
void AlignmentMuonSelectorModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("src", edm::InputTag("muons"));
  desc.add<bool>("filter", true);

  desc.add<bool>("applyBasicCuts", true);

  desc.add<double>("pMin", 0.0);
  desc.add<double>("pMax", 999999.0);
  desc.add<double>("ptMin", 10.0);
  desc.add<double>("ptMax", 999999.0);
  desc.add<double>("etaMin", -2.4);
  desc.add<double>("etaMax", 2.4);
  desc.add<double>("phiMin", -3.1416);
  desc.add<double>("phiMax", 3.1416);

  // Stand Alone Muons
  desc.add<double>("nHitMinSA", 0.0)->setComment("For Stand Alone Muons");
  desc.add<double>("nHitMaxSA", 9999999.0)->setComment("For Stand Alone Muons");
  desc.add<double>("chi2nMaxSA", 9999999.0)->setComment("For Stand Alone Muons");

  // Global Muons
  desc.add<double>("nHitMinGB", 0.0)->setComment("For Global Muons");
  desc.add<double>("nHitMaxGB", 9999999.0)->setComment("For Global Muons");
  desc.add<double>("chi2nMaxGB", 9999999.0)->setComment("For Global Muons");

  // Tracker Only
  desc.add<double>("nHitMinTO", 0.0)->setComment("For Global Muons");
  desc.add<double>("nHitMaxTO", 9999999.0)->setComment("For Global Muons");
  desc.add<double>("chi2nMaxTO", 9999999.0)->setComment("For Global Muons");

  desc.add<bool>("applyNHighestPt", false);
  desc.add<int>("nHighestPt", 2);

  desc.add<bool>("applyMultiplicityFilter", false);
  desc.add<int>("minMultiplicity", 1);

  desc.add<bool>("applyMassPairFilter", false)->setComment("copy best mass pair combination muons to result vector\nCriteria:\na) maxMassPair != minMassPair: the two highest pt muons with mass pair inside the given mass window\nb) maxMassPair == minMassPair: the muon pair with mass pair closest to given mass value");
  desc.add<double>("minMassPair", 89.0);
  desc.add<double>("maxMassPair", 90.0);

  descriptions.add("AlignmentMuonSelector", desc);
}

DEFINE_FWK_MODULE( AlignmentMuonSelectorModule );

