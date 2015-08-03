#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackReconstruction.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGeneratorFactory.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitter.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterFactory.h"

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilter.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilterFactory.h"

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleaner.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleanerFactory.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleanerWrapper.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "RecoPixelVertexing/PixelTriplets/interface/QuadrupletSeedMerger.h"

#include <vector>

//#define MKDEBUGTREE

using namespace pixeltrackfitting;
using namespace ctfseeding;
using edm::ParameterSet;

PixelTrackReconstruction::PixelTrackReconstruction(const ParameterSet& cfg,
	   edm::ConsumesCollector && iC)
  : theConfig(cfg), theFitter(0), theCleaner(0)
{
  if ( cfg.exists("SeedMergerPSet") ) {
    edm::ParameterSet mergerPSet = theConfig.getParameter<edm::ParameterSet>( "SeedMergerPSet" );
    std::string seedmergerTTRHBuilderLabel = mergerPSet.getParameter<std::string>( "ttrhBuilderLabel" );
    edm::ParameterSet seedmergerLayerList = mergerPSet.getParameter<edm::ParameterSet>( "layerList" );
    bool seedmergerAddTriplets = mergerPSet.getParameter<bool>( "addRemainingTriplets" );
    bool seedmergerMergeTriplets = mergerPSet.getParameter<bool>( "mergeTriplets" );
    theMerger_.reset(new QuadrupletSeedMerger(seedmergerLayerList, iC));
    theMerger_->setMergeTriplets( seedmergerMergeTriplets );
    theMerger_->setAddRemainingTriplets( seedmergerAddTriplets );
    theMerger_->setTTRHBuilderLabel( seedmergerTTRHBuilderLabel );
  }

  ParameterSet filterPSet = theConfig.getParameter<ParameterSet>("FilterPSet");
  std::string  filterName = filterPSet.getParameter<std::string>("ComponentName");
  if (filterName != "none") {
    theFilter.reset(PixelTrackFilterFactory::get()->create( filterName, filterPSet, iC));
    if(theConfig.exists("useFilterWithES")) {
      edm::LogInfo("Obsolete") << "useFilterWithES parameter is obsolete and can be removed";
    }
    useClusterShape = filterPSet.exists("useClusterShape");
  }

  ParameterSet orderedPSet =
      theConfig.getParameter<ParameterSet>("OrderedHitsFactoryPSet");
  std::string orderedName = orderedPSet.getParameter<std::string>("ComponentName");
  theGenerator.reset(OrderedHitsGeneratorFactory::get()->create( orderedName, orderedPSet, iC));

  ParameterSet regfactoryPSet = theConfig.getParameter<ParameterSet>("RegionFactoryPSet");
  std::string regfactoryName = regfactoryPSet.getParameter<std::string>("ComponentName");
  theRegionProducer.reset(TrackingRegionProducerFactory::get()->create(regfactoryName,regfactoryPSet, std::move(iC)));
}
  
PixelTrackReconstruction::~PixelTrackReconstruction() 
{
  halt();
}

void PixelTrackReconstruction::halt()
{
  delete theFitter; theFitter=0;
  delete theCleaner; theCleaner=0;
}

void PixelTrackReconstruction::init(const edm::EventSetup& es)
{

  ParameterSet fitterPSet = theConfig.getParameter<ParameterSet>("FitterPSet");
  std::string fitterName = fitterPSet.getParameter<std::string>("ComponentName");
  theFitter = PixelFitterFactory::get()->create( fitterName, fitterPSet);

  ParameterSet cleanerPSet = theConfig.getParameter<ParameterSet>("CleanerPSet");
  std::string  cleanerName = cleanerPSet.getParameter<std::string>("ComponentName");
  if (cleanerName != "none") theCleaner = PixelTrackCleanerFactory::get()->create( cleanerName, cleanerPSet);

  if (theMerger_) {
    theMerger_->update( es );
  }
}

void PixelTrackReconstruction::run(TracksWithTTRHs& tracks, edm::Event& ev, const edm::EventSetup& es)
{
  typedef std::vector<TrackingRegion* > Regions;
  typedef Regions::const_iterator IR;
  Regions regions = theRegionProducer->regions(ev,es);

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHand;
  es.get<TrackerTopologyRcd>().get(tTopoHand);
  const TrackerTopology *tTopo=tTopoHand.product();

  if (theFilter) theFilter->update(ev, es);

  for (IR ir=regions.begin(), irEnd=regions.end(); ir < irEnd; ++ir) {
    const TrackingRegion & region = **ir;

    const OrderedSeedingHits & triplets =  theGenerator->run(region,ev,es);
    const OrderedSeedingHits &tuplets= (theMerger_==0)? triplets : theMerger_->mergeTriplets( triplets, es );

    unsigned int nTuplets = tuplets.size();
    tracks.reserve(tracks.size()+nTuplets);
    // producing tracks
    for (unsigned int iTuplet = 0; iTuplet < nTuplets; ++iTuplet) {
      const SeedingHitSet & tuplet = tuplets[iTuplet];

      std::vector<const TrackingRecHit *> hits;
      for (unsigned int iHit = 0, nHits = tuplet.size(); iHit < nHits; ++iHit) {
        hits.push_back( tuplet[iHit]->hit() );
      }

      // fitting
      reco::Track* track = theFitter->run( ev, es, hits, region);
      if (!track) continue;

      if (theFilter) {
	if ((useClusterShape && !(*theFilter)(track, hits, tTopo)) ||
	    (!useClusterShape && !(*theFilter)(track, hits))) {
	  delete track;
	  continue;
	}
      }

      // add tracks
      tracks.push_back(TrackWithTTRHs(track, tuplet));
    }
    theGenerator->clear();
  }

  // skip ovelrapped tracks
  if (theCleaner) tracks = PixelTrackCleanerWrapper(theCleaner).clean(tracks,tTopo);

#ifdef MKDEBUGTREE2
  size_t iTrack=0;
  for(const TrackWithTTRHs& tmp: tracks) {
    const SeedingHitSet& hits = tmp.second;

    std::cout << "Track " << iTrack << " nhits " << hits.size();
    for(size_t i=0; i<hits.size(); ++i) {
      std::cout << " " << hits[i]->globalPosition();
    }
    std::cout << std::endl;

    ++iTrack;
  }
#endif

#if defined MKDEBUGTREE || defined MKDEBUGTREE2
  std::cout << "Produced " << tracks.size() << " pixel tracks" << std::endl;
#endif

  // clean memory
  for (IR ir=regions.begin(), irEnd=regions.end(); ir < irEnd; ++ir) delete (*ir);
}
