#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/VectorHitBuilderAlgorithm.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/TrackerRecHit2D/interface/VectorHit.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class VectorHitBuilderEDProducer : public edm::stream::EDProducer<>
{

 public:

  explicit VectorHitBuilderEDProducer(const edm::ParameterSet&);
  virtual ~VectorHitBuilderEDProducer();
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  void setupAlgorithm(edm::ParameterSet const& conf);
  void run(edm::Handle< edmNew::DetSetVector<Phase2TrackerCluster1D> > clusters,
           edmNew::DetSetVector<Phase2TrackerCluster1D>& clustersAcc, edmNew::DetSetVector<Phase2TrackerCluster1D>& clustersRej,
           VectorHitCollectionNew& outputAcc, VectorHitCollectionNew& outputRej);
//  VectorHitBuilderAlgorithm * algo() const { return builderAlgo; };

 private:
  VectorHitBuilderAlgorithm* builderAlgo;
  std::string offlinestubsTag;
  unsigned int maxOfflinestubs;
  std::string algoTag;
  edm::EDGetTokenT< edmNew::DetSetVector<Phase2TrackerCluster1D> > clusterProducer;
  bool readytobuild;

};

VectorHitBuilderEDProducer::VectorHitBuilderEDProducer(edm::ParameterSet const& conf)
  : offlinestubsTag( conf.getParameter<std::string>( "offlinestubs" ) ),
    maxOfflinestubs(conf.getParameter<int>( "maxVectorHits" )),
    algoTag(conf.getParameter<std::string>( "Algorithm" )),
    readytobuild(false)
{

  clusterProducer = consumes< edmNew::DetSetVector<Phase2TrackerCluster1D> >(edm::InputTag(conf.getParameter<std::string>("Clusters")));

  produces< edmNew::DetSetVector< Phase2TrackerCluster1D > >( "ClustersAccepted" );
  produces< edmNew::DetSetVector< Phase2TrackerCluster1D > >( "ClustersRejected" );
  produces< VectorHitCollectionNew >( offlinestubsTag + "Accepted" );
  produces< VectorHitCollectionNew >( offlinestubsTag + "Rejected" );
  setupAlgorithm(conf);
}

VectorHitBuilderEDProducer::~VectorHitBuilderEDProducer() {
  delete builderAlgo;
}

void VectorHitBuilderEDProducer::produce(edm::Event& event, const edm::EventSetup& es)
{
  LogDebug("VectorHitBuilder") << "VectorHitBuilderEDProducer::produce() begin";

  // get input clusters data
  edm::Handle< edmNew::DetSetVector<Phase2TrackerCluster1D> >  clustersHandle;
  event.getByToken( clusterProducer, clustersHandle);

  // create the final output collection
  std::unique_ptr< edmNew::DetSetVector< Phase2TrackerCluster1D > > outputClustersAccepted( new edmNew::DetSetVector< Phase2TrackerCluster1D > );
  std::unique_ptr< edmNew::DetSetVector< Phase2TrackerCluster1D > > outputClustersRejected( new edmNew::DetSetVector< Phase2TrackerCluster1D > );
  std::unique_ptr< VectorHitCollectionNew > outputVHAccepted( new VectorHitCollectionNew() );
  std::unique_ptr< VectorHitCollectionNew > outputVHRejected( new VectorHitCollectionNew() );

  if(readytobuild)  builderAlgo->initialize(es);
  else edm::LogError("VectorHitBuilder") << "Impossible initialization of builder!!";

  // check on the input clusters
//  builderAlgo->printClusters(*clustersHandle);

  // running the stub building algorithm
  //ERICA::output should be moved in the different algo classes?
//  run( clustersHandle, *outputClustersAccepted, *outputClustersRejected, *outputVHAccepted, *outputVHRejected);
//
//  unsigned int numberOfVectorHits = 0;
//  edmNew::DetSetVector<VectorHit>::const_iterator DSViter;
//  for( DSViter = (*outputVHAccepted).begin() ; DSViter != (*outputVHAccepted).end(); DSViter++){
//
//    edmNew::DetSet< VectorHit >::const_iterator vh;
//    for ( vh = DSViter->begin(); vh != DSViter->end(); ++vh) {
//      numberOfVectorHits++;
//      LogDebug("VectorHitBuilder") << "\t vectorhit in output " << *vh << std::endl;
//    }
//
//  }
/*
  if(numberOfVectorHits > maxOfflinestubs) {
    edm::LogError("VectorHitBuilder") <<  "Limit on the number of stubs exceeded. An empty output collection will be produced instead.\n";
    VectorHitCollectionNew empty;
    empty.swap(outputAcc);
  }
*/
  // write output to file
  event.put( std::move(outputClustersAccepted), "ClustersAccepted" );
  event.put( std::move(outputClustersRejected), "ClustersRejected" );
  event.put( std::move(outputVHAccepted), offlinestubsTag + "Accepted" );
  event.put( std::move(outputVHRejected), offlinestubsTag + "Rejected" );

//  LogDebug("VectorHitBuilder") << " Executing " << algoTag << " resulted in " << numberOfVectorHits << ".";
//  LogDebug("VectorHitBuilder") << "found\n" << numberOfVectorHits << " .\n" ;

}

void VectorHitBuilderEDProducer::setupAlgorithm(edm::ParameterSet const& conf) {

  if ( algoTag == "VectorHitBuilder" ) {
    builderAlgo = new VectorHitBuilderAlgorithm(conf);
    readytobuild = true;
  } else {
    edm::LogError("VectorHitBuilder") << " Choice " << algoTag << " is invalid.\n" ;
    readytobuild = false;
  }

}


void VectorHitBuilderEDProducer::run(edm::Handle< edmNew::DetSetVector<Phase2TrackerCluster1D> > clusters,
   edmNew::DetSetVector<Phase2TrackerCluster1D>& clustersAcc, edmNew::DetSetVector<Phase2TrackerCluster1D>& clustersRej,
   VectorHitCollectionNew& outputAcc, VectorHitCollectionNew& outputRej ){
//
//  if ( !readytobuild ) {
//    edm::LogError("VectorHitBuilder") << " No stub builder algorithm was found - cannot run!" ;
//    return;
//  }
//
//  builderAlgo->run(clusters, outputAcc, outputRej, clustersAcc, clustersRej);
//
}

#include "FWCore/Utilities/interface/typelookup.h"
TYPELOOKUP_DATA_REG(VectorHitBuilderEDProducer);
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(VectorHitBuilderEDProducer);
