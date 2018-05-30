#ifndef RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h
#define RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h

#include <mutex>
#include <utility>
#include <vector>

#include "CalibTracker/SiPixelESProducers/interface/SiPixelCPEGenericDBErrorParametrization.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "HeterogeneousCore/CUDAUtilities/interface/CUDAHostAllocator.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEBase.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelGenError.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplate.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"

#include <cuda/api_wrappers.h>

class MagneticField;
class PixelCPEFast final : public PixelCPEBase
{
public:
   struct ClusterParamGeneric : ClusterParam
   {
      ClusterParamGeneric(const SiPixelCluster & cl) : ClusterParam(cl){}
      // The truncation value pix_maximum is an angle-dependent cutoff on the
      // individual pixel signals. It should be applied to all pixels in the
      // cluster [signal_i = fminf(signal_i, pixmax)] before the column and row
      // sums are made. Morris
      int pixmx;
      
      // These are errors predicted by PIXELAV
      float sigmay; // CPE Generic y-error for multi-pixel cluster
      float sigmax; // CPE Generic x-error for multi-pixel cluster
      float sy1   ; // CPE Generic y-error for single single-pixel
      float sy2   ; // CPE Generic y-error for single double-pixel cluster
      float sx1   ; // CPE Generic x-error for single single-pixel cluster
      float sx2   ; // CPE Generic x-error for single double-pixel cluster
      
   };
   
   PixelCPEFast(edm::ParameterSet const& conf, const MagneticField *,
                   const TrackerGeometry&, const TrackerTopology&, const SiPixelLorentzAngle *,
                   const SiPixelGenErrorDBObject *, const SiPixelLorentzAngle *);
   
   
   ~PixelCPEFast();

    // The return value can only be used safely in kernels launched on
    // the same cudaStream, or after cudaStreamSynchronize.
    const pixelCPEforGPU::ParamsOnGPU *getGPUProductAsync(cuda::stream_t<>& cudaStream) const;

private:
   ClusterParam * createClusterParam(const SiPixelCluster & cl) const override;
   
   LocalPoint localPosition (DetParam const & theDetParam, ClusterParam & theClusterParam) const override;
   LocalError localError   (DetParam const & theDetParam, ClusterParam & theClusterParam) const override;
   
   static void
   collect_edge_charges(ClusterParam & theClusterParam,  //!< input, the cluster
                        int & Q_f_X,              //!< output, Q first  in X
                        int & Q_l_X,              //!< output, Q last   in X
                        int & Q_f_Y,              //!< output, Q first  in Y
                        int & Q_l_Y,              //!< output, Q last   in Y
                        bool truncate
   );
   
   
   bool UseErrorsFromTemplates_;
   bool TruncatePixelCharge_;
   
   float EdgeClusterErrorX_;
   float EdgeClusterErrorY_;
   
   std::vector<float> xerr_barrel_l1_,yerr_barrel_l1_,xerr_barrel_ln_;
   std::vector<float> yerr_barrel_ln_,xerr_endcap_,yerr_endcap_;
   float xerr_barrel_l1_def_, yerr_barrel_l1_def_,xerr_barrel_ln_def_;
   float yerr_barrel_ln_def_, xerr_endcap_def_, yerr_endcap_def_;
   
   //--- DB Error Parametrization object, new light templates 
   std::vector< SiPixelGenErrorStore > thePixelGenError_;

   std::vector<pixelCPEforGPU::DetParams, CUDAHostAllocator<pixelCPEforGPU::DetParams>> m_detParamsGPU;
   pixelCPEforGPU::CommonParams m_commonParamsGPU;     

   struct GPUDataPerDevice {
     ~GPUDataPerDevice();
     mutable std::mutex m_mutex; // protect the GPU transfer
     // not needed if not used on CPU...
     CMS_THREAD_GUARD(m_mutex) mutable pixelCPEforGPU::ParamsOnGPU h_paramsOnGPU;
     CMS_THREAD_GUARD(m_mutex) mutable pixelCPEforGPU::ParamsOnGPU * d_paramsOnGPU = nullptr;  // copy of the above on the Device
   };
   std::vector<GPUDataPerDevice> gpuDataPerDevice_;

   void fillParamsForGpu();
   void copyParamsToGpuAsync(const GPUDataPerDevice& data, cuda::stream_t<>& cudaStream) const;
};

#endif // RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h
