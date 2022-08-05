#!/bin/bash

function die { echo Failed $1: status $2 ; exit $2 ; }

TEST_DIR=${LOCALTOP}/src/HeterogeneousCore/AlpakaTest/test

if [ "x$#" != "x1" ]; then
    die "Need exactly 1 argument ('cpu', 'gpu'), got $#" 1
fi
if [ "x$1" = "xgpu" ]; then
    TARGET=gpu
elif [ "x$1" = "xcpu" ]; then
    # In non-_GPU_ IBs, if CUDA is enabled, run the GPU-targeted tests
    cudaIsEnabled
    CUDA_ENABLED=$?
    if [ "x${CUDA_ENABLED}" == "x0" ]; then
        TARGET=gpu
    else
        TARGET=cpu
    fi
else
    die "Argument needs to be 'cpu' or 'gpu', got $1" 1
fi

echo "cmsRun testAlpakaModules_cfg.py"
cmsRun ${TEST_DIR}/testAlpakaModules_cfg.py || die "cmsRun testAlpakaModules_cfg.py" $?

if [ "x${TARGET}" == "xgpu" ]; then
    echo "cmsRun testAlpakaModules_cfg.py --cuda"
    cmsRun ${TEST_DIR}/testAlpakaModules_cfg.py -- --cuda || die "cmsRun testAlpakaModules_cfg.py --cuda" $?
fi
