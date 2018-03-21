MkFit integration to CMSSW
==========================

MkFit as an external
--------------------

Build mkFit, here with CMSSW toolchain
```bash
cmsrel CMSSW_10_1_0_pre3
pushd CMSSW_10_1_0_pre3/src
cmsenv
scram setup icc-ccompiler
popd
git clone -b cmsswIntegration git@github.com:makortel/mictest.git
pushd mictest
TBB_PREFIX=$(dirname $(cd $CMSSW_BASE && scram tool tag tbb INCLUDE)) make -j 12
popd
```

Setup as external
```bash
pushd CMSSW_10_1_0_pre3/src
cat <<EOF >mkFit.xml
<tool name="mkFit" version="1.0">
  <client>
    <environment name="MKFIT_BASE" default="$PWD/../../mictest"/>
    <environment name="LIBDIR" default="$MKFIT_BASE"/>
    <environment name="INCLUDE" default="$MKFIT_BASE"/>
  </client>
  <lib name="MicCore"/>
</tool>
EOF
scram setup mkFit.xml
```