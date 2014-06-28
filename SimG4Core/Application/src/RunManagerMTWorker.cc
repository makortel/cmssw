#include "SimG4Core/Application/interface/RunManagerMTWorker.h"
#include "SimG4Core/Application/interface/RunManagerMT.h"
#include "SimG4Core/Application/interface/G4SimEvent.h"
#include "SimG4Core/Application/interface/SimRunInterface.h"
#include "SimG4Core/Application/interface/RunAction.h"
#include "SimG4Core/Application/interface/EventAction.h"
#include "SimG4Core/Application/interface/StackingAction.h"
#include "SimG4Core/Application/interface/TrackingAction.h"
#include "SimG4Core/Application/interface/SteppingAction.h"
#include "SimG4Core/Application/interface/CustomUIsession.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "SimG4Core/Notification/interface/SimActivityRegistry.h"
#include "SimG4Core/Notification/interface/SimG4Exception.h"
#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

#include "SimG4Core/Geometry/interface/DDDWorld.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "SimG4Core/Physics/interface/PhysicsList.h"

#include "SimG4Core/SensitiveDetector/interface/AttachSD.h"

#include "G4Event.hh"
#include "G4Run.hh"
#include "G4SystemOfUnits.hh"
#include "G4Threading.hh"
#include "G4UImanager.hh"
#include "G4WorkerThread.hh"
#include "G4WorkerRunManagerKernel.hh"
#include "G4StateManager.hh"

#include <atomic>
#include <thread>
#include <sstream>
#include <mutex>


// from https://hypernews.cern.ch/HyperNews/CMS/get/edmFramework/3302/2.html
namespace {
  static std::atomic<int> thread_counter{ 0 };

  int get_new_thread_index() { 
    return thread_counter++;
  }

  static thread_local int s_thread_index = get_new_thread_index();

  int getThreadIndex() { return s_thread_index; }

  void createWatchers(const edm::ParameterSet& iP,
                      SimActivityRegistry& iReg,
                      std::vector<boost::shared_ptr<SimWatcher> >& oWatchers,
                      std::vector<boost::shared_ptr<SimProducer> >& oProds
                      )
  {
    using namespace std;
    using namespace edm;
    if(!iP.exists("Watchers"))
      return;

    vector<ParameterSet> watchers = iP.getParameter<vector<ParameterSet> >("Watchers");

    for(vector<ParameterSet>::iterator itWatcher = watchers.begin();
        itWatcher != watchers.end();
        ++itWatcher) {
      std::unique_ptr<SimWatcherMakerBase> maker(
        SimWatcherFactory::get()->create(itWatcher->getParameter<std::string>("type"))
      );
      if(maker.get()==0) {
        throw SimG4Exception("Unable to find the requested Watcher");
      }

      boost::shared_ptr<SimWatcher> watcherTemp;
      boost::shared_ptr<SimProducer> producerTemp;
      maker->make(*itWatcher,iReg,watcherTemp,producerTemp);
      oWatchers.push_back(watcherTemp);
      if(producerTemp) {
        oProds.push_back(producerTemp);
      }
    }
  }
}

thread_local bool RunManagerMTWorker::m_threadInitialized = false;
thread_local bool RunManagerMTWorker::m_runTerminated = false;
thread_local std::unique_ptr<CustomUIsession> RunManagerMTWorker::m_UIsession;
thread_local RunAction *RunManagerMTWorker::m_userRunAction = nullptr;
thread_local SimRunInterface *RunManagerMTWorker::m_runInterface = nullptr;
thread_local SimActivityRegistry RunManagerMTWorker::m_registry;
thread_local SimTrackManager *RunManagerMTWorker::m_trackManager = nullptr;
thread_local std::vector<SensitiveTkDetector*> RunManagerMTWorker::m_sensTkDets;
thread_local std::vector<SensitiveCaloDetector*> RunManagerMTWorker::m_sensCaloDets;
thread_local G4Run *RunManagerMTWorker::m_currentRun = nullptr;

RunManagerMTWorker::RunManagerMTWorker(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC):
  m_generator(iConfig.getParameter<edm::ParameterSet>("Generator")),
  m_InToken(iC.consumes<edm::HepMCProduct>(iConfig.getParameter<edm::ParameterSet>("Generator").getParameter<std::string>("HepMCProductLabel"))),
  m_theLHCTlinkToken(iC.consumes<edm::LHCTransportLinkContainer>(iConfig.getParameter<edm::InputTag>("theLHCTlinkTag"))),
  m_nonBeam(iConfig.getParameter<bool>("NonBeamEvent")),
  m_EvtMgrVerbosity(iConfig.getUntrackedParameter<int>("G4EventManagerVerbosity",0)),
  m_pRunAction(iConfig.getParameter<edm::ParameterSet>("RunAction")),
  m_pEventAction(iConfig.getParameter<edm::ParameterSet>("EventAction")),
  m_pStackingAction(iConfig.getParameter<edm::ParameterSet>("StackingAction")),
  m_pTrackingAction(iConfig.getParameter<edm::ParameterSet>("TrackingAction")),
  m_pSteppingAction(iConfig.getParameter<edm::ParameterSet>("SteppingAction")),
  m_p(iConfig)
{
  edm::Service<SimActivityRegistry> otherRegistry;
  //Look for an outside SimActivityRegistry
  // this is used by the visualization code
  if(otherRegistry){
    m_registry.connect(*otherRegistry);
  }

  createWatchers(m_p, m_registry, m_watchers, m_producers);
}

RunManagerMTWorker::~RunManagerMTWorker() {
  if(!m_runTerminated) { terminateRun(); }
  // RunManagerMT has 'delete m_runInterface' in the destructor, but
  // doesn't make much sense here because it is thread_local and we're
  // not guaranteed to run the destructor on each of the threads.
}

void RunManagerMTWorker::beginRun(const RunManagerMT& runManagerMaster, const edm::EventSetup& es) {
  // Stream-specific beginRun
  // unfortunately does not work for per-thread initialization since framework does not guarantee to run them on differente threads...
  //edm::LogWarning("SimG4CoreApplication") << "RunManagerMTWorker::beginRun(): thread " << getThreadIndex();
}

void RunManagerMTWorker::endRun() {
  terminateRun();
}

void RunManagerMTWorker::initializeThread(const RunManagerMT& runManagerMaster, const edm::EventSetup& es) {
  // I guess everything initialized here should be in thread_local storage

  int thisID = getThreadIndex();

  // Initialize per-thread output
  G4Threading::G4SetThreadId( thisID );
  G4UImanager::GetUIpointer()->SetUpForAThread( thisID );
  m_UIsession.reset(new CustomUIsession());

  // Initialize worker part of shared resources (geometry, physics)
  G4WorkerThread::BuildGeometryAndPhysicsVector();

  // Create worker run manager
  G4RunManagerKernel *kernel = G4WorkerRunManagerKernel::GetRunManagerKernel();
  if(!kernel) kernel = new G4WorkerRunManagerKernel();

  // Set the geometry for the worker, share from master
  DDDWorld::WorkerSetAsWorld(runManagerMaster.world().GetWorldVolumeForWorker());

  // we need the track manager now
  m_trackManager = new SimTrackManager();

  // Get DDCompactView, or would it be better to get the object from
  // runManagerMaster instead of EventSetup in here?
  edm::ESTransientHandle<DDCompactView> pDD;
  es.get<IdealGeometryRecord>().get(pDD);

  // attach sensitive detector
  static std::mutex ddStoreMutex;
  {
    // Need to protect because DD Store is being set to read-write
    // e.g. in DDExpandedView.
    std::lock_guard<std::mutex> lk(ddStoreMutex);
    AttachSD attach;
    std::pair< std::vector<SensitiveTkDetector*>,
               std::vector<SensitiveCaloDetector*> > sensDets =
      attach.create(runManagerMaster.world(),
                    (*pDD),
                    runManagerMaster.catalog(),
                    m_p,
                    m_trackManager,
                    m_registry);

    m_sensTkDets.swap(sensDets.first);
    m_sensCaloDets.swap(sensDets.second);
  }

  edm::LogInfo("SimG4CoreApplication")
    << " RunManagerMTWorker: Sensitive Detector "
    << "building finished; found "
    << m_sensTkDets.size()
    << " Tk type Producers, and "
    << m_sensCaloDets.size()
    << " Calo type producers ";

  // Set the physics list for the worker, share from master
  PhysicsList *physicsList = runManagerMaster.physicsListForWorker();
  physicsList->InitializeWorker();
  kernel->SetPhysics(physicsList);
  kernel->InitializePhysics();

  const bool kernelInit = kernel->RunInitialization();
  if(!kernelInit)
    throw SimG4Exception("G4WorkerRunManagerKernel initialization failed");

  //tell all interesting parties that we are beginning the job
  BeginOfJob aBeginOfJob(&es);
  m_registry.beginOfJobSignal_(&aBeginOfJob);

  initializeUserActions();

  for(const std::string& command: runManagerMaster.G4Commands()) {
    edm::LogInfo("SimG4CoreApplication") << "RunManagerMTWorker:: Requests UI: "
                                         << command;
    G4UImanager::GetUIpointer()->ApplyCommand(command);
  }

  // Initialize run
  initializeRun();
}

void RunManagerMTWorker::initializeUserActions() {
  m_runInterface = new SimRunInterface(this, false);

  m_userRunAction = new RunAction(m_pRunAction, m_runInterface);
  m_userRunAction->SetMaster(false);
  Connect(m_userRunAction);

  G4RunManagerKernel *kernel = G4WorkerRunManagerKernel::GetRunManagerKernel();
  G4EventManager * eventManager = kernel->GetEventManager();
  eventManager->SetVerboseLevel(m_EvtMgrVerbosity);

  EventAction * userEventAction =
    new EventAction(m_pEventAction, m_runInterface, m_trackManager);
  Connect(userEventAction);
  eventManager->SetUserAction(userEventAction);

  TrackingAction* userTrackingAction =
    new TrackingAction(userEventAction,m_pTrackingAction);
  Connect(userTrackingAction);
  eventManager->SetUserAction(userTrackingAction);

  SteppingAction* userSteppingAction =
    new SteppingAction(userEventAction,m_pSteppingAction); 
  Connect(userSteppingAction);
  eventManager->SetUserAction(userSteppingAction);

  eventManager->SetUserAction(new StackingAction(m_pStackingAction));

}

void  RunManagerMTWorker::Connect(RunAction* runAction)
{
  runAction->m_beginOfRunSignal.connect(m_registry.beginOfRunSignal_);
  runAction->m_endOfRunSignal.connect(m_registry.endOfRunSignal_);
}

void  RunManagerMTWorker::Connect(EventAction* eventAction)
{
  eventAction->m_beginOfEventSignal.connect(m_registry.beginOfEventSignal_);
  eventAction->m_endOfEventSignal.connect(m_registry.endOfEventSignal_);
}

void  RunManagerMTWorker::Connect(TrackingAction* trackingAction)
{
  trackingAction->m_beginOfTrackSignal.connect(m_registry.beginOfTrackSignal_);
  trackingAction->m_endOfTrackSignal.connect(m_registry.endOfTrackSignal_);
}

void  RunManagerMTWorker::Connect(SteppingAction* steppingAction)
{
  steppingAction->m_g4StepSignal.connect(m_registry.g4StepSignal_);
}

void RunManagerMTWorker::initializeRun() {
  m_currentRun = new G4Run();
  G4StateManager::GetStateManager()->SetNewState(G4State_GeomClosed);
  if (m_userRunAction!=0) { m_userRunAction->BeginOfRunAction(m_currentRun); }
}

void RunManagerMTWorker::terminateRun() {
  if (m_userRunAction!=0) {
    m_userRunAction->EndOfRunAction(m_currentRun);
    delete m_userRunAction;
    m_userRunAction = nullptr;
  }

  G4RunManagerKernel *kernel = G4WorkerRunManagerKernel::GetRunManagerKernel();
  if(!kernel && !m_runTerminated) {
    m_currentEvent.reset();
    m_simEvent.reset();
    kernel->RunTermination();
    m_runTerminated = true;
  }
}

void RunManagerMTWorker::produce(const edm::Event& inpevt, const edm::EventSetup& es, const RunManagerMT& runManagerMaster) {

  if(!m_threadInitialized) {
    LogDebug("SimG4CoreApplication") << "RunManagerMTWorker::produce(): stream " << inpevt.streamID() << " thread " << getThreadIndex() << " initializing";
    initializeThread(runManagerMaster, es);
    m_threadInitialized = true;
  }
  m_runInterface->setRunManagerMTWorker(this); // For UserActions


  m_currentEvent.reset(generateEvent(inpevt));

  m_simEvent.reset(new G4SimEvent());
  m_simEvent->hepEvent(m_generator.genEvent());
  m_simEvent->weight(m_generator.eventWeight());
  if (m_generator.genVertex() !=0 ) {
    auto genVertex = m_generator.genVertex();
    m_simEvent->collisionPoint(
      math::XYZTLorentzVectorD(genVertex->x()/centimeter,
			       genVertex->y()/centimeter,
			       genVertex->z()/centimeter,
			       genVertex->t()/second));
  }
  if (m_currentEvent->GetNumberOfPrimaryVertex()==0) {
    edm::LogError("SimG4CoreApplication") 
      << " RunManagerMT::produce event " << inpevt.id().event()
      << " with no G4PrimaryVertices \n  Aborting Run" ;
       
    abortRun(false);
  } else {
    G4RunManagerKernel *kernel = G4WorkerRunManagerKernel::GetRunManagerKernel();
    if(!kernel) {
      std::stringstream ss;
      ss << "No G4WorkerRunManagerKernel yet for thread index" << getThreadIndex() << ", id " << std::hex << std::this_thread::get_id();
      throw SimG4Exception(ss.str());
    }
    kernel->GetEventManager()->ProcessOneEvent(m_currentEvent.get());
  }
    
  edm::LogInfo("SimG4CoreApplication")
    << " RunManagerMTWorker: saved : Event  " << inpevt.id().event() 
    << " stream id " << inpevt.streamID()
    << " thread index " << getThreadIndex()
    << " of weight " << m_simEvent->weight()
    << " with " << m_simEvent->nTracks() << " tracks and " 
    << m_simEvent->nVertices()
    << " vertices, generated by " << m_simEvent->nGenParts() << " particles ";
}

void RunManagerMTWorker::abortEvent() {
  if(m_runTerminated) { return; }
  G4RunManagerKernel *kernel = G4WorkerRunManagerKernel::GetRunManagerKernel();
  G4Track* t = kernel->GetEventManager()->GetTrackingManager()->GetTrack();
  t->SetTrackStatus(fStopAndKill) ;

  // CMS-specific act
  //
  TrackingAction* uta =
    static_cast<TrackingAction *>(kernel->GetEventManager()->GetUserTrackingAction());
  uta->PostUserTrackingAction(t) ;

  m_currentEvent->SetEventAborted();

  // do NOT call this method for now
  // because it'll set abortRequested=true (withing G4EventManager)
  // this will make Geant4, in the event *next* after the aborted one
  // NOT to get the primary, thus there's NOTHING to trace, and it goes
  // to the end of G4Event::DoProcessing(G4Event*), where abortRequested
  // will be reset to true again
  //
  //kernel->GetEventManager()->AbortCurrentEvent();
  //
  // instead, mimic what it does, except (re)setting abortRequested
  //
  kernel->GetEventManager()->GetStackManager()->clear() ;
  kernel->GetEventManager()->GetTrackingManager()->EventAborted() ;

  G4StateManager* stateManager = G4StateManager::GetStateManager();
  stateManager->SetNewState(G4State_GeomClosed);
}

void RunManagerMTWorker::abortRun(bool softAbort) {
  if (!softAbort) { abortEvent(); }
  delete m_currentRun;
  m_currentRun = nullptr;
  terminateRun();
}

G4Event * RunManagerMTWorker::generateEvent(const edm::Event& inpevt) {
  m_currentEvent.reset();
  m_simEvent.reset();

  G4Event * evt = new G4Event(inpevt.id().event());
  edm::Handle<edm::HepMCProduct> HepMCEvt;

  inpevt.getByToken(m_InToken, HepMCEvt);

  m_generator.setGenEvent(HepMCEvt->GetEvent());

  // required to reset the GenParticle Id for particles transported
  // along the beam pipe
  // to their original value for SimTrack creation
  resetGenParticleId( inpevt );

  if (!m_nonBeam) 
    {
      m_generator.HepMC2G4(HepMCEvt->GetEvent(),evt);
    }
  else 
    {
      m_generator.nonBeamEvent2G4(HepMCEvt->GetEvent(),evt);
    }

  return evt;
}

void RunManagerMTWorker::resetGenParticleId(const edm::Event& inpevt)
{
  edm::Handle<edm::LHCTransportLinkContainer> theLHCTlink;
  inpevt.getByToken( m_theLHCTlinkToken, theLHCTlink );
  if ( theLHCTlink.isValid() ) {
    m_trackManager->setLHCTransportLink( theLHCTlink.product() );
  }
}
