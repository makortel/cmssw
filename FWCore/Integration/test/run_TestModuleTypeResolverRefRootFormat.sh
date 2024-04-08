#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }
function runSuccess {
    echo "cmsRun $@"
    cmsRun $@ || die "cmsRun $*" $?
    echo
}

runSuccess ${SCRAM_TEST_PATH}/testModuleTypeResolverRef_cfg.py
runSuccess ${SCRAM_TEST_PATH}/testModuleTypeResolverRef_cfg.py --enableOther

runSuccess ${SCRAM_TEST_PATH}/testModuleTypeResolverRefMerge_cfg.py
runSuccess ${SCRAM_TEST_PATH}/testModuleTypeResolverRefTest_cfg.py --input moduletyperesolver_ref_merge.root
