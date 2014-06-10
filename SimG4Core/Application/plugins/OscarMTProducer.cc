#include "FWCore/PluginManager/interface/PluginManager.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimG4Core/Application/interface/OscarMTProducer.h"
#include "SimG4Core/Application/interface/G4SimEvent.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "SimG4Core/Watcher/interface/SimProducer.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "SimG4Core/Notification/interface/SimG4Exception.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/Random.h"

#include "FWCore/Concurrency/interface/SharedResourceNames.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

namespace edm {
    class StreamID;
}

namespace {
    //
    // this machinery allows to set CLHEP static engine
    // to the one defined by RandomNumberGenerator service
    // at the beginning of an event, and reset it back to
    // "default-default" at the end of the event;
    // Dave D. has decided to implement it this way because
    // we don't know if there're other modules using CLHEP
    // static engine, thus we want to ensure that the one
    // we use for OscarMTProducer is unique to OscarMTProducer
    //
    // !!! This not only sets the random engine used by GEANT.
    // There are a few SimWatchers/SimProducers that generate
    // random number and also use the global CLHEP random engine
    // set by this code. If we ever change this design be careful
    // not to forget about them!!!

    class StaticRandomEngineSetUnset {
    public:
        StaticRandomEngineSetUnset(edm::StreamID const&);
        explicit StaticRandomEngineSetUnset(CLHEP::HepRandomEngine * engine);
        ~StaticRandomEngineSetUnset();
        CLHEP::HepRandomEngine* getEngine() const;
    private:
        CLHEP::HepRandomEngine* m_currentEngine;
        CLHEP::HepRandomEngine* m_previousEngine;
    };
}

OscarMTMasterThread::OscarMTMasterThread(std::shared_ptr<RunManagerMT> runManager, const edm::EventSetup& iSetup):
//OscarMTMasterThread::OscarMTMasterThread(const edm::ParameterSet& iConfig, const edm::EventSetup& iSetup):
  m_runManager(runManager)
{

  const edm::ParameterSet& pset = m_runManager->parameterSet();
  SimActivityRegistry *registry = m_runManager->registry(); // must be done in the current thread
  RunManagerMT::ESProducts esprod = m_runManager->readES(iSetup);
  //m_runManager->readES(iSetup);

#define MK_THREADS
#ifdef MK_THREADS
  // Lock the mutex
  std::unique_lock<std::mutex> lk(m_startMutex);

  edm::LogWarning("Test") << "Main thread, address " << esprod.pDD;

  // Create thread
  m_masterThread = std::thread([&](){
#endif
      std::shared_ptr<RunManagerMTMaster> runManagerMaster;
#ifdef MK_THREADS
      {
        // Lock the mutex (i.e. wait until the creating thread has called cv.wait()
        std::lock_guard<std::mutex> lk2(m_startMutex);
#endif
        runManagerMaster = std::make_shared<RunManagerMTMaster>(pset, registry);
        //auto runManager = std::make_shared<RunManagerMT>(iConfig);
        m_runManagerMaster = runManagerMaster;
        runManagerMaster->initG4(esprod.pDD, esprod.pMF, esprod.pTable, iSetup);
#ifdef MK_THREADS
        edm::LogWarning("Test") << "Master thread, address " << esprod.pDD;
      }
      // G4 initialization finish, send signal to the other thread to continue
      m_startCanProceed = true;
      m_startCv.notify_one();
      edm::LogWarning("Test") << "Master thread, notified main thread";

      // Lock the other mutex, and wait a signal via the condition variable
      std::unique_lock<std::mutex> lk2(m_stopMutex);
      edm::LogWarning("Test") << "Master thread, locked mutex, starting wait";
      m_stopCanProceed = false;
      m_stopCv.wait(lk2, [&](){return m_stopCanProceed;});

      // Then do clean-up
      edm::LogWarning("Test") << "Master thread, woke up, starting cleanup";
#endif
      runManagerMaster->stopG4();
      edm::LogWarning("Test") << "Master thread, stopped G4, am I unique owner?" << runManagerMaster.unique();
      runManagerMaster.reset(); // must be done in this thread, segfault otherwise
#ifdef MK_THREADS
      edm::LogWarning("Test") << "Master thread, reseted shared_ptr";
      lk2.unlock();
      edm::LogWarning("Test") << "Master thread, finished";
    });
  //m_runManager->initG4(iSetup);

  // Start waiting a signal from the condition variable (releases the lock temporarily)

  m_startCanProceed = false;
  m_startCv.wait(lk, [&](){return m_startCanProceed;});
  // Unlock the lock
  lk.unlock();
#endif
  edm::LogWarning("Test") << "Main thread, again address " << esprod.pDD;
}

OscarMTMasterThread::~OscarMTMasterThread() {
#ifdef MK_THREADS
  edm::LogWarning("Test") << "Main thread, destructor";
  {
    std::lock_guard<std::mutex> lk(m_stopMutex);
#endif
    m_runManagerMaster.reset();
#ifdef MK_THREADS
    edm::LogWarning("Test") << "Main thread, reseted shared_ptr";
  }
  edm::LogWarning("Test") << "Main thread, going to signal master thread";
  m_stopCanProceed = true;
  m_stopCv.notify_one();
  edm::LogWarning("Test") << "Main thread, going to join master thread";
  m_masterThread.join();
  edm::LogWarning("Test") << "Main thread, finished";
#endif
}

OscarMTProducer::OscarMTProducer(edm::ParameterSet const & p, const edm::ParameterSet *)
{
#ifdef MK_SERIAL
  // Random number generation not allowed here
  StaticRandomEngineSetUnset random(nullptr);

  usesResource(edm::SharedResourceNames::kGEANT);
  usesResource(edm::SharedResourceNames::kCLHEPRandomEngine);

  consumes<edm::HepMCProduct>(p.getParameter<edm::InputTag>("HepMCProductLabel"));
  m_runManager.reset(new RunManagerMT(p));
  //m_runManager.reset(new RunManagerMT(p, consumesCollector()));

  produces<edm::SimTrackContainer>().setBranchAlias("SimTracks");
  produces<edm::SimVertexContainer>().setBranchAlias("SimVertices");
  produces<edm::PSimHitContainer>("TrackerHitsPixelBarrelLowTof");
  produces<edm::PSimHitContainer>("TrackerHitsPixelBarrelHighTof");
  produces<edm::PSimHitContainer>("TrackerHitsTIBLowTof");
  produces<edm::PSimHitContainer>("TrackerHitsTIBHighTof");
  produces<edm::PSimHitContainer>("TrackerHitsTIDLowTof");
  produces<edm::PSimHitContainer>("TrackerHitsTIDHighTof");
  produces<edm::PSimHitContainer>("TrackerHitsPixelEndcapLowTof");
  produces<edm::PSimHitContainer>("TrackerHitsPixelEndcapHighTof");
  produces<edm::PSimHitContainer>("TrackerHitsTOBLowTof");
  produces<edm::PSimHitContainer>("TrackerHitsTOBHighTof");
  produces<edm::PSimHitContainer>("TrackerHitsTECLowTof");
  produces<edm::PSimHitContainer>("TrackerHitsTECHighTof");
    
  produces<edm::PSimHitContainer>("TotemHitsT1");
  produces<edm::PSimHitContainer>("TotemHitsT2Gem");
  produces<edm::PSimHitContainer>("TotemHitsRP");
  produces<edm::PSimHitContainer>("FP420SI");
  produces<edm::PSimHitContainer>("BSCHits");
  produces<edm::PSimHitContainer>("PLTHits");
  produces<edm::PSimHitContainer>("BCM1FHits");

  produces<edm::PCaloHitContainer>("EcalHitsEB");
  produces<edm::PCaloHitContainer>("EcalHitsEE");
  produces<edm::PCaloHitContainer>("EcalHitsES");
  produces<edm::PCaloHitContainer>("HcalHits");
  produces<edm::PCaloHitContainer>("CaloHitsTk");
  produces<edm::PSimHitContainer>("MuonDTHits");
  produces<edm::PSimHitContainer>("MuonCSCHits");
  produces<edm::PSimHitContainer>("MuonRPCHits");
  produces<edm::PSimHitContainer>("MuonGEMHits");
  produces<edm::PCaloHitContainer>("CastorPL");
  produces<edm::PCaloHitContainer>("CastorFI");
  produces<edm::PCaloHitContainer>("CastorBU");
  produces<edm::PCaloHitContainer>("CastorTU");
  produces<edm::PCaloHitContainer>("EcalTBH4BeamHits");
  produces<edm::PCaloHitContainer>("HcalTB06BeamHits");
  produces<edm::PCaloHitContainer>("ZDCHITS"); 
  produces<edm::PCaloHitContainer>("ChamberHits"); 
  produces<edm::PCaloHitContainer>("FibreHits"); 
  produces<edm::PCaloHitContainer>("WedgeHits"); 
    
  //register any products 
  m_producers = m_runManager->producers();

  for(Producers::iterator itProd = m_producers.begin();
      itProd != m_producers.end(); ++itProd) {

    (*itProd)->registerProducts(*this);
  }

  //UIsession manager for message handling
  m_UIsession.reset(new CustomUIsession());
#endif
}

OscarMTProducer::~OscarMTProducer() 
{ }

std::unique_ptr<edm::ParameterSet> OscarMTProducer::initializeGlobalCache(const edm::ParameterSet& iConfig) {
  return std::unique_ptr<edm::ParameterSet>(new edm::ParameterSet(iConfig));
}

std::shared_ptr<OscarMTMasterThread> OscarMTProducer::globalBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup, const edm::ParameterSet *iConfig) {
  auto runManager = std::make_shared<RunManagerMT>(*iConfig);
  auto masterThread = std::make_shared<OscarMTMasterThread>(runManager, iSetup);
  return masterThread;
}

void OscarMTProducer::globalEndRun(const edm::Run& iRun, const edm::EventSetup& iSetup, const RunContext *iContext) {

}

void OscarMTProducer::globalEndJob(edm::ParameterSet *iConfig) {
}

void 
OscarMTProducer::beginRun(const edm::Run & r, const edm::EventSetup & es)
{
#ifdef MK_SERIAL
  // Random number generation not allowed here
  StaticRandomEngineSetUnset random(nullptr);
  m_runManager->initG4(es);
#endif
}

void 
OscarMTProducer::endRun(const edm::Run&, const edm::EventSetup&)
{
#ifdef MK_SERIAL
  m_runManager->stopG4();
#endif
}

void OscarMTProducer::produce(edm::Event & e, const edm::EventSetup & es)
{
#ifdef MK_SERIAL
  StaticRandomEngineSetUnset random(e.streamID());

  std::vector<SensitiveTkDetector*>& sTk = 
    m_runManager->sensTkDetectors();
  std::vector<SensitiveCaloDetector*>& sCalo =
    m_runManager->sensCaloDetectors();

  try {

    m_runManager->produce(e, es);

    std::auto_ptr<edm::SimTrackContainer> 
      p1(new edm::SimTrackContainer);
    std::auto_ptr<edm::SimVertexContainer> 
      p2(new edm::SimVertexContainer);
    G4SimEvent * evt = m_runManager->simEvent();
    evt->load(*p1);
    evt->load(*p2);   

    e.put(p1);
    e.put(p2);

    for (std::vector<SensitiveTkDetector*>::iterator it = sTk.begin(); 
	 it != sTk.end(); ++it) {

      std::vector<std::string> v = (*it)->getNames();
      for (std::vector<std::string>::iterator in = v.begin(); 
	   in!= v.end(); ++in) {

	std::auto_ptr<edm::PSimHitContainer> 
	  product(new edm::PSimHitContainer);
	(*it)->fillHits(*product,*in);
	e.put(product,*in);
      }
    }
    for (std::vector<SensitiveCaloDetector*>::iterator it = sCalo.begin(); 
	 it != sCalo.end(); ++it) {

      std::vector<std::string>  v = (*it)->getNames();

      for (std::vector<std::string>::iterator in = v.begin(); 
	   in!= v.end(); in++) {

	std::auto_ptr<edm::PCaloHitContainer> 
	  product(new edm::PCaloHitContainer);
	(*it)->fillHits(*product,*in);
	e.put(product,*in);
      }
    }

    for(Producers::iterator itProd = m_producers.begin();
	itProd != m_producers.end(); ++itProd) {

      (*itProd)->produce(e,es);
    }

  } catch ( const SimG4Exception& simg4ex ) {
       
    edm::LogInfo("SimG4CoreApplication") << " SimG4Exception caght !" 
					 << simg4ex.what();
       
    m_runManager->abortEvent();
    throw edm::Exception( edm::errors::EventCorruption );
  }
#endif
}

StaticRandomEngineSetUnset::StaticRandomEngineSetUnset(
      edm::StreamID const& streamID)
{
  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable()) {
    throw cms::Exception("Configuration")
      << "The OscarMTProducer module requires the RandomNumberGeneratorService\n"
      "which is not present in the configuration file.  You must add the service\n"
      "in the configuration file if you want to run OscarMTProducer";
  }
  m_currentEngine = &(rng->getEngine(streamID));

  m_previousEngine = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(m_currentEngine);
}

StaticRandomEngineSetUnset::StaticRandomEngineSetUnset(
      CLHEP::HepRandomEngine * engine) 
{
  m_currentEngine = engine;
  m_previousEngine = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(m_currentEngine);
}

StaticRandomEngineSetUnset::~StaticRandomEngineSetUnset() 
{
  CLHEP::HepRandom::setTheEngine(m_previousEngine);
}

CLHEP::HepRandomEngine* StaticRandomEngineSetUnset::getEngine() const 
{ 
  return m_currentEngine; 
}

DEFINE_FWK_MODULE(OscarMTProducer);
