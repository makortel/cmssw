#include "SimG4Core/Application/interface/RunManagerMT.h"

#include "SimG4Core/Notification/interface/SimActivityRegistry.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

#include "HepPDT/ParticleDataTable.hh"
#include "SimGeneral/HepPDTRecord/interface/PDTRecord.h"


RunManagerMT::RunManagerMT(const edm::ParameterSet& iConfig):
  m_p(iConfig),
  firstRun(true),
  m_pUseMagneticField(iConfig.getParameter<bool>("UseMagneticField"))
  /*
  m_nonBeam(iConfig.getParameter<bool>("NonBeamEvent")),
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

RunManagerMT::ESProducts RunManagerMT::readES(const edm::EventSetup& iSetup) {
  bool geomChanged = idealGeomRcdWatcher_.check(iSetup);
  if (geomChanged && (!firstRun)) {
    throw cms::Exception("BadConfig") 
      << "[SimG4Core RunManagerMT]\n"
      << "The Geometry configuration is changed during the job execution\n"
      << "this is not allowed, the geometry must stay unchanged\n";
  }
  if (m_pUseMagneticField) {
    bool magChanged = idealMagRcdWatcher_.check(iSetup);
    if (magChanged && (!firstRun)) {
      throw cms::Exception("BadConfig") 
	<< "[SimG4Core RunManagerMT]\n"
	<< "The MagneticField configuration is changed during the job execution\n"
	<< "this is not allowed, the MagneticField must stay unchanged\n";
    }
  }

  ESProducts ret;

  // DDDWorld: get the DDCV from the ES and use it to build the World
  edm::ESTransientHandle<DDCompactView> pDD;
  iSetup.get<IdealGeometryRecord>().get(pDD);
  ret.pDD = pDD.product();

  if(m_pUseMagneticField) {
    edm::ESHandle<MagneticField> pMF;
    iSetup.get<IdealMagneticFieldRecord>().get(pMF);
    ret.pMF = pMF.product();
  }

  edm::ESHandle<HepPDT::ParticleDataTable> fTable;
  iSetup.get<PDTRecord>().get(fTable);
  ret.pTable = fTable.product();

  firstRun = false;
  return ret;
}
