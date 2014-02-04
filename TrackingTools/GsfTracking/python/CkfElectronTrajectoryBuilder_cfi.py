import FWCore.ParameterSet.Config as cms

import copy
import RecoTracker.CkfPattern.CkfTrajectoryBuilder_cfi
CkfElectronTrajectoryBuilder = copy.deepcopy(RecoTracker.CkfPattern.CkfTrajectoryBuilder_cfi.CkfTrajectoryBuilder)
CkfElectronTrajectoryBuilder.propagatorAlong = 'fwdElectronPropagator'
CkfElectronTrajectoryBuilder.propagatorOpposite = 'bwdElectronPropagator'
CkfElectronTrajectoryBuilder.estimator = 'electronChi2'

