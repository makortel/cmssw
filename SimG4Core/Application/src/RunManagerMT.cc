#include "SimG4Core/Application/interface/RunManagerMT.h"

#include "SimG4Core/Notification/interface/SimActivityRegistry.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

RunManagerMT::RunManagerMT(const edm::ParameterSet& iConfig):
  m_p(iConfig)
  /*
  m_nonBeam(iConfig.getParameter<bool>("NonBeamEvent")),
  m_pUseMagneticField(iConfig.getParameter<bool>("UseMagneticField")),
  m_PhysicsTablesDir(iConfig.getParameter<std::string>("PhysicsTablesDirectory")),
  m_StorePhysicsTables(iConfig.getParameter<bool>("StorePhysicsTables")),
  m_RestorePhysicsTables(iConfig.getParameter<bool>("RestorePhysicsTables")),
  m_EvtMgrVerbosity(iConfig.getUntrackedParameter<int>("G4EventManagerVerbosity",0)),
  m_pField(iConfig.getParameter<edm::ParameterSet>("MagneticField")),
  m_pGenerator(iConfig.getParameter<edm::ParameterSet>("Generator")),
  m_pPhysics(iConfig.getParameter<edm::ParameterSet>("Physics")),
  m_pRunAction(iConfig.getParameter<edm::ParameterSet>("RunAction")),
  m_pEventAction(iConfig.getParameter<edm::ParameterSet>("EventAction")),
  m_pStackingAction(iConfig.getParameter<edm::ParameterSet>("StackingAction")),
  m_pTrackingAction(iConfig.getParameter<edm::ParameterSet>("TrackingAction")),
  m_pSteppingAction(iConfig.getParameter<edm::ParameterSet>("SteppingAction")),
  m_G4Commands(iConfig.getParameter<std::vector<std::string> >("G4Commands")),
//m_p(p)
  m_theLHCTlinkTag(iConfig.getParameter<edm::InputTag>("theLHCTlinkTag"))
  */
{
}

RunManagerMT::~RunManagerMT() {}

SimActivityRegistry *RunManagerMT::registry() {
  edm::Service<SimActivityRegistry> otherRegistry;
  if(otherRegistry)
    return otherRegistry.operator->();
  return nullptr;
}
