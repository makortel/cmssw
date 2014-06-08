#ifndef SimG4Core_RunManagerMT_H
#define SimG4Core_RunManagerMT_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include <vector>

class SimActivityRegistry;

class RunManagerMT {
public:
  RunManagerMT(const edm::ParameterSet& iConfig);
  ~RunManagerMT();

  const edm::ParameterSet& parameterSet() const { return m_p; }
  SimActivityRegistry *registry();

private:
  edm::ParameterSet m_p;

  /*
  const bool m_nonBeam;
  const bool m_pUseMagneticField;
  const std::string m_PhysicsTablesDir;
  const bool m_StorePhysicsTables;
  const bool m_RestorePhysicsTables;
  const int m_EvtMgrVerbosity;
  const edm::ParameterSet m_pField;
  const edm::ParameterSet m_pGenerator;
  const edm::ParameterSet m_pPhysics;
  const edm::ParameterSet m_pRunAction;
  const edm::ParameterSet m_pEventAction;
  const edm::ParameterSet m_pStackingAction;
  const edm::ParameterSet m_pTrackingAction;
  const edm::ParameterSet m_pSteppingAction;
  const std::vector<std::string> m_G4Commands;
  //edm::ParameterSet m_p;

  edm::InputTag m_theLHCTlinkTag;
  */
};

#endif
