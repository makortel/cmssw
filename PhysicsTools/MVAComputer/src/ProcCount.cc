// -*- C++ -*-
//
// Package:     MVAComputer
// Class  :     ProcCount
//

// Implementation:
//     Variable processor that returns the number of input variable instances
//
// Author:      Christophe Saout
// Created:     Fri May 18 20:05 CEST 2007
//

#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/MVAComputer/interface/VarProcessor.h"
#include "PhysicsTools/MVAComputer/interface/Calibration.h"

using namespace PhysicsTools;

namespace {  // anonymous

  class ProcCount : public VarProcessor {
  public:
    typedef VarProcessor::Registry::Registry<ProcCount, Calibration::ProcCount> Registry;

    ProcCount(const char *name, const Calibration::ProcCount *calib, const MVAComputer *computer);
    ~ProcCount() override {}

    void configure(ConfIterator iter, unsigned int n) override;
    void eval(ValueIterator iter, unsigned int n) const override;
  };

  ProcCount::Registry registry("ProcCount");

  ProcCount::ProcCount(const char *name, const Calibration::ProcCount *calib, const MVAComputer *computer)
      : VarProcessor(name, calib, computer) {}

  void ProcCount::configure(ConfIterator iter, unsigned int n) {
    while (iter)
      iter++(Variable::FLAG_ALL) << Variable::FLAG_NONE;
  }

  void ProcCount::eval(ValueIterator iter, unsigned int n) const {
    while (iter) {
      unsigned int count = iter.size();
      iter(count);
      iter++;
    }
  }

}  // anonymous namespace
