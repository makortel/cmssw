#ifndef SimG4Core_RunManagerMT_H
#define SimG4Core_RunManagerMT_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include <string>
#include <vector>

class SimActivityRegistry;

class DDCompactView;
class MagneticField;

namespace HepPDT {
  class ParticleDataTable;
}

class RunManagerMT {
public:
  RunManagerMT(const edm::ParameterSet& iConfig);
  ~RunManagerMT();

  const edm::ParameterSet& parameterSet() const { return m_p; }
  SimActivityRegistry *registry();

  struct ESProducts {
    ESProducts(): pDD(nullptr), pMF(nullptr), pTable(nullptr) {}
    const DDCompactView *pDD;
    const MagneticField *pMF;
    const HepPDT::ParticleDataTable *pTable;
  };
  ESProducts readES(const edm::EventSetup& iSetup);

private:
  edm::ParameterSet m_p;

  bool firstRun;

  edm::ESWatcher<IdealGeometryRecord> idealGeomRcdWatcher_;
  edm::ESWatcher<IdealMagneticFieldRecord> idealMagRcdWatcher_;

  const bool m_pUseMagneticField;
  /*
  const bool m_nonBeam;
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
