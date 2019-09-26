Minimal instructions
```
# Input data for the simulation
nvprof -o profile.nvvp cmsRun profile_daq.py
./HeterogeneousCore/CUDATest/test/extract-nvvp.py profile.nvvp --skipEvents 2 --maxEvents 4000 # creates config.json

# Calibration of CUDA timing
pushd HeterogeneousCore/CUDATest/test
./cudaCalibration.py
popd

# Test simulation
cmsRun HeterogeneousCore/CUDATest/test/testSimulationPatatrack_cfg.py

# Run simulation scan
./HeterogeneousCore/CUDATest/test/runSimulation.py ./HeterogeneousCore/CUDATest/test/testSimulationPatatrack_cfg.py
```
