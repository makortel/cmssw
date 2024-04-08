#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }
function runSuccess {
    echo "cmsRun $@"
    cmsRun $@ || die "cmsRun $*" $?
    echo
}
function runFailure {
    echo "cmsRun $@ (exepcted to fail)"
    cmsRun $@ && die "cmsRun $*" 1
    echo
}

runSuccess ${SCRAM_TEST_PATH}/testRefProductIDMetadataConsistencyStreamer_cfg.py
runSuccess ${SCRAM_TEST_PATH}/testRefProductIDMetadataConsistencyStreamer_cfg.py --enableOther

runSuccess ${SCRAM_TEST_PATH}/testRefProductIDMetadataConsistencyStreamerTest_cfg.py --input refconsistency_1.dat --input refconsistency_10.dat

echo "Concatenating streamer files"
CatStreamerFiles refconsistency_cat.dat refconsistency_1.dat refconsistency_10.dat
echo

runFailure ${SCRAM_TEST_PATH}/testModuleTypeResolverRefTest_cfg.py --input moduletyperesolver_ref_cat.dat
