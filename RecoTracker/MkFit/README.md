# MkFit integration to CMSSW

## MkFit as an external

### Build mkFit

Here with CMSSW toolchain

```bash
cmsrel CMSSW_10_2_0_pre3
pushd CMSSW_10_2_0_pre3/src
cmsenv
git cms-init
scram setup icc-ccompiler
source $(scram tool tag icc-ccompiler ICC_CCOMPILER_BASE)/bin/iccvars.sh intel64
popd
git clone -b cmsswIntegration git@github.com:makortel/mictest.git
pushd mictest
TBB_PREFIX=$(dirname $(cd $CMSSW_BASE && scram tool tag tbb INCLUDE)) make -j 12
popd
```

### Setup as external

```bash
pushd CMSSW_10_2_0_pre3/src
cat <<EOF >mkfit.xml
<tool name="mkfit" version="1.0">
  <client>
    <environment name="MKFITBASE" default="$PWD/../../mictest"/>
    <environment name="LIBDIR" default="\$MKFITBASE/lib"/>
    <environment name="INCLUDE" default="\$MKFITBASE"/>
  </client>
  <runtime name="MKFIT_BASE" value="\$MKFITBASE"/>
  <lib name="MicCore"/>
  <lib name="MkFit"/>
</tool>
EOF
scram setup mkfit.xml
cmsenv
```

### Obtain CMSSW code and build

```bash
# in CMSSW_10_2_0_pre3/src
git cms-remote add makortel
git fetch makortel
git checkout -b mkfit_1020pre3_tsg makortel/mkfit_1020pre3_tsg
git cms-addpkg $(git diff $CMSSW_VERSION --name-only | cut -d/ -f-2 | uniq)
git cms-checkdeps -a
scram b -j <N>
```

## Using MkFit in InitialStep

Example below uses 2018 trackingOnly workflow
```bash
# Generate configuration
runTheMatrix.py -l 10824.1 --apply 2 --command "--customise RecoTracker/MkFit/customizeInitialStepToMkFit.customizeInitialStepToMkFit" -j 0
cd 10824.1*
# edit step3*RECO*.py to contain your desired (2018 RelVal MC) input files
cmsRun step3*RECO*.py
```
