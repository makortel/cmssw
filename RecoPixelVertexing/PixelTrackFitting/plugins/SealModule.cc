#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "PixelTrackProducer.h"
DEFINE_FWK_MODULE(PixelTrackProducer);

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterBase.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterFactory.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterByConformalMappingAndLine.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterByHelixProjections.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/KFBasedPixelFitter.h"
DEFINE_EDM_PLUGIN(PixelFitterFactory, PixelFitterByConformalMappingAndLine, "PixelFitterByConformalMappingAndLine");
DEFINE_EDM_PLUGIN(PixelFitterFactory, PixelFitterByHelixProjections, "PixelFitterByHelixProjections");
 DEFINE_EDM_PLUGIN(PixelFitterFactory, KFBasedPixelFitter, "KFBasedPixelFitter");

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleaner.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleanerFactory.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleanerBySharedHits.h"
DEFINE_EDM_PLUGIN(PixelTrackCleanerFactory, PixelTrackCleanerBySharedHits, "PixelTrackCleanerBySharedHits");
