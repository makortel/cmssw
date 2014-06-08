#ifndef SimG4Core_RunManagerMT_H
#define SimG4Core_RunManagerMT_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class RunManagerMT {
public:
  RunManagerMT(const edm::ParameterSet& iConfig);
  ~RunManagerMT();
private:
};

#endif
