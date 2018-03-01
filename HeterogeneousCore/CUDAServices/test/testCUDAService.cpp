#include <cassert>
#include <iostream>
#include <string>
#include <utility>

#include <cuda_runtime_api.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

int main()
{
  int rc = -1;
  try
  {
    edm::ActivityRegistry ar;
    int deviceCount = 0;
    auto ret = cudaGetDeviceCount( &deviceCount );
    if( ret != cudaSuccess )
      throw cms::Exception("Unable to query the CUDA capable devices from the CUDA runtime API.");

    // Enable the service only if there are GPU capable devices installed
    bool configEnabled( deviceCount );
    if( !configEnabled )
      std::cout << "No CUDA capable devices found." << std::endl;
    edm::ParameterSet ps;
    ps.addUntrackedParameter( "enabled", configEnabled );

    CUDAService cs(ps, ar);

    // Test that the service is enabled
    assert( cs.enabled() == configEnabled );
    std::cout << "The CUDAService is enabled." << std::endl;

    // At this point, we can get, as info, the driver and runtime versions.
    int driverVersion = 0, runtimeVersion = 0;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    std::cout << "CUDA Driver Version / Runtime Version: " << driverVersion/1000 << "." << (driverVersion%100)/10 
	      << " / " << runtimeVersion/1000 << "." << (runtimeVersion%100)/10 << std::endl;

    // Test that the number of devices found by the service
    // is the same as detected by the CUDA runtime API
    assert( cs.numberOfDevices() == deviceCount );
    std::cout << "Detected " << cs.numberOfDevices() << " CUDA Capable device(s)" << std::endl;

    // Test that the compute capabilities of each device
    // are the same as detected by the CUDA runtime API
    for( int i=0; i<deviceCount; ++i )
    {
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, i);
      assert(deviceProp.major == cs.computeCapability(i).first);
      assert(deviceProp.minor == cs.computeCapability(i).second);
      std::cout << "Device " << i << ": " << deviceProp.name
		<< "\n CUDA Capability Major/Minor version number: " << deviceProp.major << "." << deviceProp.minor
		<< std::endl;
      std::cout << std::endl;
    }

    // Now forcing the service to be disabled...
    edm::ParameterSet psf;
    configEnabled = false;
    psf.addUntrackedParameter( "enabled", configEnabled );
    CUDAService csf(psf, ar);
    std::cout << "CUDAService disabled by configuration" << std::endl;

    // Test that the service is actually disabled
    assert( csf.enabled() == configEnabled );

    //Fake the end-of-job signal.
    ar.postEndJobSignal_();
    rc = 0;
  }
  catch( cms::Exception & exc )
  {
    std::cerr << exc << std::endl;
    rc = 1;
  }
  catch( ... )
  {
    std::cerr << "Unknown exception caught" << std::endl;
    rc = 2;
  }
  return rc;
}
