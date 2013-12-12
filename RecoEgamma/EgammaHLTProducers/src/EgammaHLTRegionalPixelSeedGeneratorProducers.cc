//
// Package:         RecoEgamma/EgammaHLTProducers
// Class:           EgammaHLTRegionalPixelSeedGeneratorProducers
//  Modified from TkSeedGeneratorFromTrk by Jeremy Werner, Princeton University, USA
// $Id: EgammaHLTRegionalPixelSeedGeneratorProducers.cc,v 1.13 2012/01/23 12:56:38 sharper Exp $
//

#include <iostream>
#include <memory>
#include <string>

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTRegionalPixelSeedGeneratorProducers.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGeneratorFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromRegionHits.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedCreatorFactory.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Math/interface/Point3D.h"
// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

using namespace std;
using namespace reco;

EgammaHLTRegionalPixelSeedGeneratorProducers::EgammaHLTRegionalPixelSeedGeneratorProducers(edm::ParameterSet const& conf) :   conf_(conf),combinatorialSeedGenerator(0)
{

  produces<TrajectorySeedCollection>();

  ptmin_       = conf_.getParameter<double>("ptMin");
  vertexz_     = conf_.getParameter<double>("vertexZ");
  originradius_= conf_.getParameter<double>("originRadius");
  halflength_  = conf_.getParameter<double>("originHalfLength");
  deltaEta_    = conf_.getParameter<double>("deltaEtaRegion");
  deltaPhi_    = conf_.getParameter<double>("deltaPhiRegion");

  candTag_     = consumes<reco::RecoEcalCandidateCollection>(conf_.getParameter< edm::InputTag > ("candTag"));
  candTagEle_  = consumes<reco::ElectronCollection>(conf_.getParameter< edm::InputTag > ("candTagEle"));
  BSProducer_  = consumes<reco::BeamSpot>(conf.getParameter<edm::InputTag>("BSProducer"));
  
  useZvertex_  = conf_.getParameter<bool>("UseZInVertex");

  // setup orderedhits setup (in order to tell seed generator to use pairs/triplets, which layers)
}

// Virtual destructor needed.
EgammaHLTRegionalPixelSeedGeneratorProducers::~EgammaHLTRegionalPixelSeedGeneratorProducers() { 
}

void EgammaHLTRegionalPixelSeedGeneratorProducers::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;
  desc.add<double>("ptMin", 1.5);
  desc.add<double>("vertexZ", 0);
  desc.add<double>("originRadius", 0.02);
  desc.add<double>("originHalfLength", 15.0);
  desc.add<double>("deltaEtaRegion", 0.3);
  desc.add<double>("deltaPhiRegion", 0.3);
  desc.add<edm::InputTag>(("candTag"), edm::InputTag("hltL1SeededRecoEcalCandidate"));
  desc.add<edm::InputTag>(("candTagEle"), edm::InputTag("pixelMatchElectrons"));
  desc.add<edm::InputTag>(("BSProducer"), edm::InputTag("hltOnlineBeamSpot"));
  desc.add<bool>(("UseZInVertex"), false);
  desc.add<std::string>("TTRHBuilder", "WithTrackAngle");

  edm::ParameterSetDescription orederedHitsPSET;
  orederedHitsPSET.add<std::string>("ComponentName", "StandardHitPairGenerator");
  orederedHitsPSET.add<std::string>("SeedingLayers", "hltESPPixelLayerPairs");
  orederedHitsPSET.add<unsigned int>("maxElement", 0);
  desc.add<edm::ParameterSetDescription>("OrderedHitsFactoryPSet", orederedHitsPSET);

  descriptions.add(("hltEgammaHLTRegionalPixelSeedGeneratorProducers"), desc);  
}


// Functions that gets called by framework every event
void EgammaHLTRegionalPixelSeedGeneratorProducers::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // resulting collection
  std::auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection());    

  // Get the recoEcalCandidates
  edm::Handle<reco::RecoEcalCandidateCollection> recoecalcands;
  iEvent.getByToken(candTag_,recoecalcands);

  //Get the Beam Spot position
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByToken(BSProducer_,recoBeamSpotHandle);
  // gets its position
  const BeamSpot::Point& BSPosition = recoBeamSpotHandle->position(); 

  //Get the HLT electrons collection if needed
  edm::Handle<reco::ElectronCollection> electronHandle;
  if(useZvertex_)
    iEvent.getByToken(candTagEle_,electronHandle);

  reco::SuperClusterRef scRef;
  for (reco::RecoEcalCandidateCollection::const_iterator recoecalcand= recoecalcands->begin(); recoecalcand!=recoecalcands->end(); recoecalcand++) {
    scRef = recoecalcand->superCluster();
    float zvertex = 0;
    if( useZvertex_ ){
      reco::SuperClusterRef scRefEle;
      for(reco::ElectronCollection::const_iterator iElectron = electronHandle->begin(); iElectron != electronHandle->end(); iElectron++){
	//Compare electron SC with EcalCandidate SC
	scRefEle = iElectron->superCluster();
	if(&(*scRef) == &(*scRefEle)){
	  if(iElectron->track().isNonnull()) zvertex = iElectron->track()->vz();
	  else  zvertex = iElectron->gsfTrack()->vz();
	  break;
	}
      }

    }
    GlobalVector dirVector((recoecalcand)->px(),(recoecalcand)->py(),(recoecalcand)->pz());
    RectangularEtaPhiTrackingRegion etaphiRegion( dirVector,
											   GlobalPoint( BSPosition.x(), BSPosition.y(), zvertex ), 
											   ptmin_,
											   originradius_,
											   halflength_,
											   deltaEta_,
											   deltaPhi_);

    // fill Trajectory seed collection
    combinatorialSeedGenerator->run(*output, etaphiRegion, iEvent, iSetup);
    
  }

    iEvent.put(output);
}
