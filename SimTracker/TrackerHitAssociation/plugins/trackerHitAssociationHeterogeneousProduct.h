#ifndef SimTrackerTrackerHitAssociationClusterHeterogeneousProduct_H
#define SimTrackerTrackerHitAssociationClusterHeterogeneousProduct_H


namespace trackerHitAssociationHeterogeneousProduct {

  struct CPUProduct {
    ClusterTPAssociation collection;
  }

  struct ClusterSLGPU {

   ClusterSLGPU * me_d=nullptr;
   std::array<uint32_t,4> * links_d;
   uint32_t * tkId_d;
   uint32_t * tkId2_d;
   uint32_t * n1_d;
   uint32_t * n2_d;

   static constexpr uint32_t MAX_DIGIS = 2000*150;
   static constexpr uint32_t MaxNumModules = 2000;

  };

  struct GPUProduct {
     ClusterSLGPU *  gpu_d=nullptr;
  };

   using ClusterTPAHeterogeneousProduct = HeterogeneousProductImpl<heterogeneous::CPUProduct<CPUProduct>,
                                                                   heterogeneous::GPUCudaProduct<GPUProduct> >;

}

#endif
