#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }
function runSuccess {
    echo "cmsRun $@"
    cmsRun $@ || die "cmsRun $*" $?
    echo
}

runSuccess ${SCRAM_TEST_PATH}/testModuleTypeResolverRef_cfg.py --streamer
runSuccess ${SCRAM_TEST_PATH}/testModuleTypeResolverRef_cfg.py --streamer --enableOther

runSuccess ${SCRAM_TEST_PATH}/testModuleTypeResolverRefTest_cfg.py --input moduletyperesolver_ref_1.dat --input moduletyperesolver_ref_10.dat

echo "Concatenating streamer files"
CatStreamerFiles moduletyperesolver_ref_cat.dat moduletyperesolver_ref_1.dat moduletyperesolver_ref_10.dat
echo

runSuccess ${SCRAM_TEST_PATH}/testModuleTypeResolverRefTest_cfg.py --input moduletyperesolver_ref_cat.dat
