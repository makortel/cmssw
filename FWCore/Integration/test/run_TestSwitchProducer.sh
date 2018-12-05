#!/bin/bash

test=testSwitchProducer

function die { echo Failure $1: status $2 ; exit $2 ; }

pushd ${LOCAL_TMP_DIR}

  echo "SwitchProducer in a Task"
  cmsRun ${LOCAL_TEST_DIR}/${test}Task_cfg.py > testSwitchProducerTask1.log 2>/dev/null || die "cmsRun ${test}Task_cfg.py 1" $?
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/testSwitchProducerTask1.log testSwitchProducerTask1.log || die "comparing testSwitchProducerTask1.log" $?

  echo "SwitchProducer in a Task, case test2 disabled"
  cmsRun ${LOCAL_TEST_DIR}/${test}Task_cfg.py disableTest2 > testSwitchProducerTask2.log 2>/dev/null || die "cmsRun ${test}Task_cfg.py 2" $?
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/testSwitchProducerTask2.log testSwitchProducerTask2.log || die "comparing testSwitchProducerTask2.log" $?

  echo "Merge outputs"
  edmCopyPickMerge outputFile=testSwitchProducerMerge1.root inputFiles=file:testSwitchProducerTask1.root inputFiles=file:testSwitchProducerTask2.root > testSwitchProducerMerge1.log 2>/dev/null || die "edmCopyPickMerge 1" $?
  echo "Merge outputs in reverse order"
  edmCopyPickMerge outputFile=testSwitchProducerMerge2.root inputFiles=file:testSwitchProducerTask2.root inputFiles=file:testSwitchProducerTask1.root > testSwitchProducerMerge2.log 2>/dev/null || die "edmCopyPickMerge 2" $?

  echo "Test provenance of merged output"
  cmsRun ${LOCAL_TEST_DIR}/${test}ProvenanceAnalyzer_cfg.py testSwitchProducerMerge1.root > testSwitchProducerProvenanceAnalyzer1.log 2>/dev/null || die "cmsRun ${test}ProvenanceAnalyzer_cfg.py 1" $?
  echo "Test provenance of reversely merged output"
  cmsRun ${LOCAL_TEST_DIR}/${test}ProvenanceAnalyzer_cfg.py testSwitchProducerMerge2.root > testSwitchProducerProvenanceAnalyzer2.log 2>/dev/null || die "cmsRun ${test}ProvenanceAnalyzer_cfg.py 2" $?

  
  echo "SwitchProducer in a Path"
  cmsRun ${LOCAL_TEST_DIR}/${test}Path_cfg.py > testSwitchProducerPath1.log 2>/dev/null || die "cmsRun ${test}Path_cfg.py 1" $?
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/testSwitchProducerPath1.log testSwitchProducerPath1.log || die "comparing testSwitchProducerPath1.log" $?

  echo "SwitchProducer in a Path, case test2 disabled"
  cmsRun ${LOCAL_TEST_DIR}/${test}Path_cfg.py disableTest2 > testSwitchProducerPath2.log 2>/dev/null || die "cmsRun ${test}Path_cfg.py 2" $?
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/testSwitchProducerPath2.log testSwitchProducerPath2.log || die "comparing testSwitchProducerPath2.log" $?


  echo "SwitchProducer in a Path after a failing filter"
  cmsRun ${LOCAL_TEST_DIR}/${test}PathFilter_cfg.py > testSwitchProducerPathFilter1.log 2>/dev/null || die "cmsRun ${test}PathFilter_cfg.py 1" $?
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/testSwitchProducerPathFilter1.log testSwitchProducerPathFilter1.log || die "comparing testSwitchProducerPathFilter1.log" $?

  echo "SwitchProducer in a Path after a failing filter, case test2 disabled"
  cmsRun ${LOCAL_TEST_DIR}/${test}PathFilter_cfg.py disableTest2 > testSwitchProducerPathFilter2.log 2>/dev/null || die "cmsRun ${test}PathFilter_cfg.py 2" $?
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/testSwitchProducerPathFilter2.log testSwitchProducerPathFilter2.log || die "comparing testSwitchProducerPathFilter2.log" $?

popd

exit 0
