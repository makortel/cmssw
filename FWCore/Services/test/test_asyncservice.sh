#!/bin/bash

function die { cat log.txt; echo $1: status $2 ;  exit $2; }

CONF=${SCRAM_TEST_PATH}/test_asyncservice_cfg.py
echo "cmsRun ${CONF}"
cmsRun ${CONF} > log.txt 2>&1 || die "Failure using ${CONF}" $?

echo "cmsRun ${CONF} --exception"
cmsRun ${CONF} --exception > log.txt 2>&1
RET=$?
if [ "${RET}" == "0" ]; then
    cat log.txt
    die "${CONF} --exception succeeded while it was expected to fail" 1
fi
grep -q "ZombieKillerService" log.txt && die "${CONF} --exception was killed by ZombieKillerService, while the job should have failed by itself" 1
grep -q "AsyncCallNotAllowed" log.txt || die "${CONF} --exception did not fail with AsyncCallNotAllowed" $?
grep -q "Framework is shutting down, further run() calls are not allowed" log.txt || die "${CONF} --exception did not contain expected exception message" $?
