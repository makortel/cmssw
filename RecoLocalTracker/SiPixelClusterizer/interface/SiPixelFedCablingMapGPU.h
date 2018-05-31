#ifndef RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPU_h
#define RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPU_h

// TODO: since this has more information than just cabling map, maybe we should invent a better name?
struct SiPixelFedCablingMapGPU {
  unsigned int size = 0;
  unsigned int * fed = nullptr;
  unsigned int * link = nullptr;
  unsigned int * roc = nullptr;
  unsigned int * RawId = nullptr;
  unsigned int * rocInDet = nullptr;
  unsigned int * moduleId = nullptr;
  unsigned char * badRocs = nullptr;
};

#endif
