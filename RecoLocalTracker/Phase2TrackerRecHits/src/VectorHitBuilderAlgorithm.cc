#include "RecoLocalTracker/Phase2TrackerRecHits/interface/VectorHitBuilderAlgorithm.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/VectorHit2D.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/VectorHitBuilderAlgorithm.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoLocalTracker/Records/interface/TkPhase2OTCPERecord.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPE.h"

#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"

VectorHitBuilderAlgorithm::VectorHitBuilderAlgorithm(const edm::ParameterSet& conf) :
  nMaxVHforeachStack(conf.getParameter<int>("maxVectorHitsinaStack")),
  barrelCut(conf.getParameter< std::vector< double > >("BarrelCut")),
  endcapCut(conf.getParameter< std::vector< double > >("EndcapCut")),
  cpeTag_(conf.getParameter<edm::ESInputTag>("CPE")),
  theFitter(new LinearFit())
{}

void VectorHitBuilderAlgorithm::initialize(const edm::EventSetup& es)
{
  //FIXME:ask Vincenzo
  /*
  uint32_t tk_cache_id = es.get<TrackerDigiGeometryRecord>().cacheIdentifier();
  uint32_t c_cache_id = es.get<TkPhase2OTCPERecord>().cacheIdentifier();
  
  if(tk_cache_id != tracker_cache_id) {
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  tracker_cache_id = tk_cache_id;
  }
  if(c_cache_id != cpe_cache_id) {
  es.get<TkPhase2OTCPERecord>().get(matcherTag, matcher);
  es.get<TkPhase2OTCPERecord>().get(cpeTag, cpe);
  cpe_cache_id = c_cache_id;
  }
  */

  // get the geometry and topology
  edm::ESHandle< TrackerGeometry > geomHandle;
  es.get< TrackerDigiGeometryRecord >().get( geomHandle );
  initTkGeom(geomHandle);

  edm::ESHandle< TrackerTopology > tTopoHandle;
  es.get< TrackerTopologyRcd >().get(tTopoHandle);
  initTkTopo(tTopoHandle);

  // load the cpe via the eventsetup
  edm::ESHandle<ClusterParameterEstimator<Phase2TrackerCluster1D> > cpeHandle;
  es.get<TkPhase2OTCPERecord>().get(cpeTag_, cpeHandle);
  initCpe(cpeHandle.product());
}

void VectorHitBuilderAlgorithm::initTkGeom(edm::ESHandle< TrackerGeometry > tkGeomHandle){
  theTkGeom = tkGeomHandle.product();
}
void VectorHitBuilderAlgorithm::initTkTopo(edm::ESHandle< TrackerTopology > tkTopoHandle){
  theTkTopo = tkTopoHandle.product();
}
void VectorHitBuilderAlgorithm::initCpe(const ClusterParameterEstimator<Phase2TrackerCluster1D>* cpeProd){
  cpe = cpeProd;
}

double VectorHitBuilderAlgorithm::computeParallaxCorrection(const PixelGeomDetUnit*& geomDetUnit_low, const Point3DBase<float, LocalTag>& lPosClu_low,
                                                            const PixelGeomDetUnit*& geomDetUnit_upp, const Point3DBase<float, LocalTag>& lPosClu_upp){
  double parallCorr = 0.0;
  Global3DPoint origin(0,0,0);
  Global3DPoint gPosClu_low = geomDetUnit_low->surface().toGlobal(lPosClu_low);
  GlobalVector gV = gPosClu_low - origin;
  LogTrace("VectorHitBuilderAlgorithm") << " global vector passing to the origin:" << gV;

  LocalVector lV = geomDetUnit_low->surface().toLocal(gV);
  LogTrace("VectorHitBuilderAlgorithm") << " local vector passing to the origin (in low sor):" << lV;
  LocalVector lV_norm = lV/lV.z();
  LogTrace("VectorHitBuilderAlgorithm") << " normalized local vector passing to the origin (in low sor):" << lV_norm;

  Global3DPoint gPosClu_upp = geomDetUnit_upp->surface().toGlobal(lPosClu_upp);
  Local3DPoint lPosClu_uppInLow = geomDetUnit_low->surface().toLocal(gPosClu_upp);
  parallCorr = lV_norm.x() * lPosClu_uppInLow.z();

  return parallCorr;
}

void VectorHitBuilderAlgorithm::printClusters(const edmNew::DetSetVector<Phase2TrackerCluster1D>& clusters){

  int nCluster = 0;
  int numberOfDSV = 0;
  edmNew::DetSetVector<Phase2TrackerCluster1D>::const_iterator DSViter;
  for( DSViter = clusters.begin() ; DSViter != clusters.end(); DSViter++){

    ++numberOfDSV;

    // Loop over the clusters in the detector unit
    for (edmNew::DetSet< Phase2TrackerCluster1D >::const_iterator clustIt = DSViter->begin(); clustIt != DSViter->end(); ++clustIt) {

      nCluster++;

      // get the detector unit's id
      const GeomDetUnit* geomDetUnit(theTkGeom->idToDetUnit(DSViter->detId()));
      if (!geomDetUnit) return;

      printCluster(geomDetUnit, clustIt);

    }
  }
  LogDebug("VectorHitBuilderAlgorithm") << " Number of input clusters: " << nCluster << std::endl;

}

void VectorHitBuilderAlgorithm::printCluster(const GeomDet* geomDetUnit, const Phase2TrackerCluster1D* clustIt){

  if (!geomDetUnit) return;
  const PixelGeomDetUnit* pixelGeomDetUnit = dynamic_cast< const PixelGeomDetUnit* >(geomDetUnit);
  const PixelTopology& topol = pixelGeomDetUnit->specificTopology();
  if (!pixelGeomDetUnit) return;

  unsigned int layer = theTkTopo->layer(geomDetUnit->geographicalId());
  unsigned int module = theTkTopo->module(geomDetUnit->geographicalId());
  LogTrace("VectorHitBuilderAlgorithm") << "Layer:" << layer << " and DetId: " << geomDetUnit->geographicalId().rawId() << std::endl;
  TrackerGeometry::ModuleType mType = theTkGeom->getDetectorType(geomDetUnit->geographicalId());
  if (mType == TrackerGeometry::ModuleType::Ph2PSP)
    LogTrace("VectorHitBuilderAlgorithm") << "Pixel cluster (module:" << module << ") " << std::endl;
  else if (mType == TrackerGeometry::ModuleType::Ph2SS || mType == TrackerGeometry::ModuleType::Ph2PSS)
    LogTrace("VectorHitBuilderAlgorithm") << "Strip cluster (module:" << module << ") " << std::endl;
  else LogTrace("VectorHitBuilderAlgorithm") << "no module?!" << std::endl;
  LogTrace("VectorHitBuilderAlgorithm") << "with pitch:" << topol.pitch().first << " , " << topol.pitch().second << std::endl;
  LogTrace("VectorHitBuilderAlgorithm") << " and width:" << pixelGeomDetUnit->surface().bounds().width() << " , lenght:" << pixelGeomDetUnit->surface().bounds().length() << std::endl;


  auto && lparams = cpe->localParameters( *clustIt, *pixelGeomDetUnit );
  Global3DPoint gparams = pixelGeomDetUnit->surface().toGlobal(lparams.first);

  LogTrace("VectorHitBuilderAlgorithm") << "\t global pos " << gparams << std::endl;
  LogTrace("VectorHitBuilderAlgorithm") << "\t local  pos " << lparams.first << "with err " << lparams.second << std::endl;
  LogTrace("VectorHitBuilderAlgorithm") << std::endl;

  return;
}

void VectorHitBuilderAlgorithm::loadDetSetVector( std::map< DetId,std::vector<VectorHit> >& theMap, edmNew::DetSetVector<VectorHit>& theCollection ) const{

  std::map<DetId,std::vector<VectorHit> >::const_iterator it = theMap.begin();
  std::map<DetId,std::vector<VectorHit> >::const_iterator lastDet = theMap.end();
  for( ; it != lastDet ; ++it ) {
    edmNew::DetSetVector<VectorHit>::FastFiller vh_col(theCollection, it->first);
    std::vector<VectorHit>::const_iterator vh_it = it->second.begin();
    std::vector<VectorHit>::const_iterator vh_end = it->second.end();
    for( ; vh_it != vh_end ; ++vh_it)  {
      vh_col.push_back(*vh_it);
    }
  }

}

void VectorHitBuilderAlgorithm::run(edm::Handle< edmNew::DetSetVector<Phase2TrackerCluster1D> > clusters, 
                                    VectorHitCollectionNew& vhAcc, 
                                    VectorHitCollectionNew& vhRej, 
                                    edmNew::DetSetVector<Phase2TrackerCluster1D>& clustersAcc, 
                                    edmNew::DetSetVector<Phase2TrackerCluster1D>& clustersRej)
{

  LogDebug("VectorHitBuilderAlgorithm") << "Run VectorHitBuilderAlgorithm ... \n" ;
  const  edmNew::DetSetVector<Phase2TrackerCluster1D>* ClustersPhase2Collection = clusters.product();


  std::map< DetId, std::vector<VectorHit> > tempVHAcc, tempVHRej;
  std::map< DetId, std::vector<VectorHit> >::iterator it_temporary;

  //loop over the DetSetVector
  LogDebug("VectorHitBuilderAlgorithm") << "with #clusters : " << ClustersPhase2Collection->size() << std::endl ;
  for( auto DSViter : *ClustersPhase2Collection){

    unsigned int rawDetId1(DSViter.detId());
    DetId detId1(rawDetId1);
    DetId lowerDetId, upperDetId;
    if( theTkTopo->isLower(detId1) ){
      lowerDetId = detId1;
      upperDetId = theTkTopo->partnerDetId(detId1);
    } else if (theTkTopo->isUpper(detId1)) {
      upperDetId = detId1;
      lowerDetId = theTkTopo->partnerDetId(detId1);
    }
    DetId detIdStack = theTkTopo->stack(detId1);

    //debug
    LogDebug("VectorHitBuilderAlgorithm") << "  DetId stack : " << detIdStack.rawId() << std::endl;
    LogDebug("VectorHitBuilderAlgorithm") << "  DetId lower set of clusters  : " << lowerDetId.rawId();
    LogDebug("VectorHitBuilderAlgorithm") << "  DetId upper set of clusters  : " << upperDetId.rawId() << std::endl;

    it_temporary = tempVHAcc.find(detIdStack);
    if ( it_temporary != tempVHAcc.end() ) {
      LogTrace("VectorHitBuilderAlgorithm") << " this stack has already been analyzed -> skip it ";
      continue;
    }

    const GeomDet* gd;
    const StackGeomDet* stackDet;
    edmNew::DetSetVector<Phase2TrackerCluster1D>::const_iterator it_detLower = ClustersPhase2Collection->find( lowerDetId );
    edmNew::DetSetVector<Phase2TrackerCluster1D>::const_iterator it_detUpper = ClustersPhase2Collection->find( upperDetId );

    if ( it_detLower != ClustersPhase2Collection->end() && it_detUpper != ClustersPhase2Collection->end() ){

      gd = theTkGeom->idToDet(detIdStack);
      stackDet = dynamic_cast<const StackGeomDet*>(gd);
      std::vector<VectorHit> vhsInStack_Acc;
      std::vector<VectorHit> vhsInStack_Rej; 
      const auto vhsInStack_AccRej = buildVectorHits(stackDet, clusters, *it_detLower, *it_detUpper);

      //storing accepted and rejected VHs
      for(auto vh : vhsInStack_AccRej ) {
        if(vh.second == true){
          vhsInStack_Acc.push_back(vh.first);
        }
        else if(vh.second == false){
          vhsInStack_Rej.push_back(vh.first);
        }
      }

      //ERICA:: to be checked with map!
      //sorting vhs for best chi2
      std::sort(vhsInStack_Acc.begin(), vhsInStack_Acc.end());

      tempVHAcc[detIdStack] = vhsInStack_Acc;
      tempVHRej[detIdStack] = vhsInStack_Rej;

      LogTrace("VectorHitBuilderAlgorithm") << "For detId #" << detIdStack.rawId() << " the following VHits have been accepted:";
      for (auto vhIt : vhsInStack_Acc){
        LogTrace("VectorHitBuilderAlgorithm") << "accepted VH: " << vhIt;
      }
      LogTrace("VectorHitBuilderAlgorithm") << "For detId #" << detIdStack.rawId() << " the following VHits have been rejected:";
      for (auto vhIt : vhsInStack_Rej){
        LogTrace("VectorHitBuilderAlgorithm") << "rejected VH: " << vhIt;
      }
    
    }

  }

  loadDetSetVector(tempVHAcc, vhAcc);
  loadDetSetVector(tempVHRej, vhRej);

  LogDebug("VectorHitBuilderAlgorithm") << "End run VectorHitBuilderAlgorithm ... \n" ;
  return;

}

bool VectorHitBuilderAlgorithm::LocalPositionSort::operator()(Phase2TrackerCluster1DRef clus1, Phase2TrackerCluster1DRef clus2) const
{
  const PixelGeomDetUnit* gdu1 = dynamic_cast< const PixelGeomDetUnit* >(geomDet_);
  auto && lparams1 = cpe_->localParameters( *clus1, *gdu1 );          // x, y, z, e2_xx, e2_xy, e2_yy
  auto && lparams2 = cpe_->localParameters( *clus2, *gdu1 );          // x, y, z, e2_xx, e2_xy, e2_yy
  return lparams1.first.x() < lparams2.first.x();
}

bool VectorHitBuilderAlgorithm::checkClustersCompatibilityBeforeBuilding(edm::Handle< edmNew::DetSetVector<Phase2TrackerCluster1D> > clusters,
                                                                         const detset & theLowerDetSet,
                                                                         const detset & theUpperDetSet)
{
  if(theLowerDetSet.size()==1 && theUpperDetSet.size()==1) return true;

  //order lower clusters in u
  std::vector<Phase2TrackerCluster1D> lowerClusters;
  if(theLowerDetSet.size()>1) LogDebug("VectorHitBuilderAlgorithm") << " more than 1 lower cluster! " << std::endl;
  if(theUpperDetSet.size()>1) LogDebug("VectorHitBuilderAlgorithm") << " more than 1 upper cluster! " << std::endl;
  for ( const_iterator cil = theLowerDetSet.begin(); cil != theLowerDetSet.end(); ++ cil ) {
    Phase2TrackerCluster1DRef clusterLower = edmNew::makeRefTo( clusters, cil );
    lowerClusters.push_back(*clusterLower);
  }
  return true;
}

bool VectorHitBuilderAlgorithm::checkClustersCompatibility(Local3DPoint& poslower, 
                                                           Local3DPoint& posupper, 
                                                           LocalError& errlower, 
                                                           LocalError& errupper)
{

  return true;

}

//----------------------------------------------------------------------------
//ERICA::in the DT code the global position is used to compute the alpha angle and put a cut on that.
std::vector<std::pair<VectorHit,bool>> VectorHitBuilderAlgorithm::buildVectorHits(const StackGeomDet * stack, 
                                                                  edm::Handle< edmNew::DetSetVector<Phase2TrackerCluster1D> > clusters, 
                                                                  const detset & theLowerDetSet, 
                                                                  const detset & theUpperDetSet,
                                                                  const std::vector<bool>& phase2OTClustersToSkip)
{

  std::vector<std::pair<VectorHit,bool>> result;
  if(checkClustersCompatibilityBeforeBuilding(clusters, theLowerDetSet, theUpperDetSet)){
    LogDebug("VectorHitBuilderAlgorithm") << "  compatible -> continue ... " << std::endl;
  } else { LogTrace("VectorHitBuilderAlgorithm") << "  not compatible, going to the next cluster"; }

  std::vector<Phase2TrackerCluster1DRef> lowerClusters;
  for ( const_iterator cil = theLowerDetSet.begin(); cil != theLowerDetSet.end(); ++ cil ) {
    Phase2TrackerCluster1DRef clusterLower = edmNew::makeRefTo( clusters, cil );
    lowerClusters.push_back(clusterLower);
  }
  std::vector<Phase2TrackerCluster1DRef> upperClusters;
  for ( const_iterator ciu = theUpperDetSet.begin(); ciu != theUpperDetSet.end(); ++ ciu ) {
    Phase2TrackerCluster1DRef clusterUpper = edmNew::makeRefTo( clusters, ciu );
    upperClusters.push_back(clusterUpper);
  }

  std::sort(lowerClusters.begin(), lowerClusters.end(), LocalPositionSort(&*theTkGeom,&*cpe,&*stack->lowerDet()));
  std::sort(upperClusters.begin(), upperClusters.end(), LocalPositionSort(&*theTkGeom,&*cpe,&*stack->upperDet()));
  
  for ( auto cluL : lowerClusters){
    LogDebug("VectorHitBuilderAlgorithm") << " lower clusters " << std::endl;
    printCluster(stack->lowerDet(),&*cluL);
    const PixelGeomDetUnit* gduLow = dynamic_cast< const PixelGeomDetUnit* >(stack->lowerDet());
    auto && lparamsLow = cpe->localParameters( *cluL, *gduLow );
    for ( auto cluU : upperClusters){
      LogDebug("VectorHitBuilderAlgorithm") << "\t upper clusters " << std::endl;
      printCluster(stack->upperDet(),&*cluU);
      const PixelGeomDetUnit* gduUpp = dynamic_cast< const PixelGeomDetUnit* >(stack->upperDet());
      auto && lparamsUpp = cpe->localParameters( *cluU, *gduUpp );

      //applying the parallax correction
      double pC = computeParallaxCorrection(gduLow,lparamsLow.first,gduUpp,lparamsUpp.first);
      LogDebug("VectorHitBuilderAlgorithm") << " \t parallax correction:" << pC << std::endl;
      double lpos_upp_corr = 0.0;
      double lpos_low_corr = 0.0;
      if(lparamsUpp.first.x() > lparamsLow.first.x()){
        if(lparamsUpp.first.x() > 0){
          lpos_low_corr = lparamsLow.first.x();
          lpos_upp_corr = lparamsUpp.first.x() - fabs(pC);
        }
        if(lparamsUpp.first.x() < 0){
          lpos_low_corr = lparamsLow.first.x() + fabs(pC);
          lpos_upp_corr = lparamsUpp.first.x();
        }
      } else if( lparamsUpp.first.x() < lparamsLow.first.x() ) {
        if(lparamsUpp.first.x() > 0){
          lpos_low_corr = lparamsLow.first.x() - fabs(pC);
          lpos_upp_corr = lparamsUpp.first.x();
        }
        if(lparamsUpp.first.x() < 0){
          lpos_low_corr = lparamsLow.first.x();
          lpos_upp_corr = lparamsUpp.first.x() + fabs(pC);
        }
      } else {
        if(lparamsUpp.first.x() > 0){
          lpos_low_corr = lparamsLow.first.x();
          lpos_upp_corr = lparamsUpp.first.x() - fabs(pC);
        }
        if(lparamsUpp.first.x() < 0){
          lpos_low_corr = lparamsLow.first.x();
          lpos_upp_corr = lparamsUpp.first.x() + fabs(pC);
        }
      }

      LogDebug("VectorHitBuilderAlgorithm") << " \t local pos upper corrected (x):" << lpos_upp_corr << std::endl;
      LogDebug("VectorHitBuilderAlgorithm") << " \t local pos lower corrected (x):" << lpos_low_corr << std::endl;

      //building my tolerance : 10*sigma
      double delta = 10.0*sqrt(lparamsLow.second.xx()+lparamsUpp.second.xx()); 
      LogDebug("VectorHitBuilderAlgorithm") << " \t delta: " << delta << std::endl;

      double width = lpos_low_corr - lpos_upp_corr;
      LogDebug("VectorHitBuilderAlgorithm") << " \t width: " << width << std::endl;


      unsigned int layerStack = theTkTopo->layer(stack->geographicalId());
      if(stack->subDetector() == GeomDetEnumerators::SubDetector::P2OTB ) LogDebug("VectorHitBuilderAlgorithm") << " \t is barrel.    " << std::endl;
      if(stack->subDetector() == GeomDetEnumerators::SubDetector::P2OTEC) LogDebug("VectorHitBuilderAlgorithm") << " \t is endcap.    " << std::endl;
      LogDebug("VectorHitBuilderAlgorithm") << " \t layer is : " << layerStack << std::endl;

      float cut = 0.0;
      if(stack->subDetector() == GeomDetEnumerators::SubDetector::P2OTB ) cut = barrelCut.at(layerStack);
      if(stack->subDetector() == GeomDetEnumerators::SubDetector::P2OTEC) cut = endcapCut.at(layerStack);
      LogDebug("VectorHitBuilderAlgorithm") << " \t the cut is:" << cut << std::endl;

      //no cut
      LogDebug("VectorHitBuilderAlgorithm") << " accepting VH! " << std::endl;
      VectorHit vh = buildVectorHit( stack, cluL, cluU);
      if (vh.isValid()){
        result.push_back(std::make_pair(vh, true));
      }

/*
      //old cut: indipendent from layer
      //if( (lpos_upp_corr < lpos_low_corr + delta) && 
      //    (lpos_upp_corr > lpos_low_corr - delta) ){
      //new cut: dependent on layers
      if(fabs(width) < cut){
        LogDebug("VectorHitBuilderAlgorithm") << " accepting VH! " << std::endl;
        VectorHit vh = buildVectorHit( stack, cluL, cluU);
        //protection: the VH can also be empty!!
        if (vh.isValid()){
          result.push_back(std::make_pair(vh, true));
        }

      } else {
        LogDebug("VectorHitBuilderAlgorithm") << " rejecting VH: " << std::endl;
        //storing vh rejected for combinatiorial studies
        VectorHit vh = buildVectorHit( stack, cluL, cluU);
        result.push_back(std::make_pair(vh, false));
      }
*/
      
    }
  }





/*
  for ( const_iterator cil = theLowerDetSet.begin(); cil != theLowerDetSet.end(); ++ cil ) {
    //possibility to introducing the skipping of the clusters
    //if(phase2OTClustersToSkip.empty() or (not phase2OTClustersToSkip[cil]) ) {

    Phase2TrackerCluster1DRef clusterLower = edmNew::makeRefTo( clusters, cil );

    for ( const_iterator ciu = theUpperDetSet.begin(); ciu != theUpperDetSet.end(); ++ ciu ) {

      LogTrace("VectorHitBuilderAlgorithm")<<" in the loop for upper clusters with index " << ciu << " on detId " << stack->geographicalId().rawId();

      Phase2TrackerCluster1DRef clusterUpper = edmNew::makeRefTo( clusters, ciu );
      VectorHit vh = buildVectorHit( stack, clusterLower, clusterUpper);
      LogTrace("VectorHitBuilderAlgorithm") << "-> Vectorhit " << vh ;
      LogTrace("VectorHitBuilderAlgorithm") << std::endl;
      //protection: the VH can also be empty!!

      if (vh.isValid()){
        result.push_back(vh);
      }

    }
  }
*/

  //if( result.size() > nMaxVHforeachStack ){
  //  result.erase(result.begin()+nMaxVHforeachStack, result.end());
  //}

  return result;

}

VectorHit VectorHitBuilderAlgorithm::buildVectorHit(const StackGeomDet * stack, 
                                                    Phase2TrackerCluster1DRef lower, 
                                                    Phase2TrackerCluster1DRef upper)
{

  LogTrace("VectorHitBuilderAlgorithm") << "Build VH with: ";
  //printCluster(stack->lowerDet(),&*lower);
  //printCluster(stack->upperDet(),&*upper);

  const PixelGeomDetUnit* geomDetLower = dynamic_cast< const PixelGeomDetUnit* >(stack->lowerDet());
  const PixelGeomDetUnit* geomDetUpper = dynamic_cast< const PixelGeomDetUnit* >(stack->upperDet());

  auto && lparamsLower = cpe->localParameters( *lower, *geomDetLower );          // x, y, z, e2_xx, e2_xy, e2_yy
  Global3DPoint gparamsLower = geomDetLower->surface().toGlobal(lparamsLower.first);
  LogTrace("VectorHitBuilderAlgorithm") << "\t lower global pos: " << gparamsLower ;

  auto && lparamsUpper = cpe->localParameters( *upper, *geomDetUpper );
  Global3DPoint gparamsUpper = geomDetUpper->surface().toGlobal(lparamsUpper.first);
  LogTrace("VectorHitBuilderAlgorithm") << "\t upper global pos: " << gparamsUpper ;

  //local parameters of upper cluster in lower system of reference
  Local3DPoint lparamsUpperInLower = geomDetLower->surface().toLocal(gparamsUpper);

  LogTrace("VectorHitBuilderAlgorithm") << "\t lower global pos: " << gparamsLower ;
  LogTrace("VectorHitBuilderAlgorithm") << "\t upper global pos: " << gparamsUpper ;

  LogTrace("VectorHitBuilderAlgorithm") << "A:\t lower local pos: " << lparamsLower.first << " with error: " << lparamsLower.second << std::endl;
  LogTrace("VectorHitBuilderAlgorithm") << "A:\t upper local pos in the lower sof " << lparamsUpperInLower << " with error: " << lparamsUpper.second << std::endl;

  bool ok = checkClustersCompatibility(lparamsLower.first, lparamsUpper.first, lparamsLower.second, lparamsUpper.second);

  if(ok){

    AlgebraicSymMatrix22 covMat2Dzx;
    double chi22Dzx = 0.0;
    Local3DPoint pos2Dzx;
    Local3DVector dir2Dzx;
    fit2Dzx(lparamsLower.first, lparamsUpperInLower, lparamsLower.second, lparamsUpper.second, pos2Dzx, dir2Dzx, covMat2Dzx, chi22Dzx);
    LogTrace("VectorHitBuilderAlgorithm") << "\t  pos2Dzx: " << pos2Dzx;
    LogTrace("VectorHitBuilderAlgorithm") << "\t  dir2Dzx: " << dir2Dzx;
    LogTrace("VectorHitBuilderAlgorithm") << "\t  cov2Dzx: " << covMat2Dzx;
    VectorHit2D vh2Dzx = VectorHit2D(pos2Dzx, dir2Dzx, covMat2Dzx, chi22Dzx);

    AlgebraicSymMatrix22 covMat2Dzy;
    double chi22Dzy = 0.0;
    Local3DPoint pos2Dzy;
    Local3DVector dir2Dzy;
    fit2Dzy(lparamsLower.first, lparamsUpperInLower, lparamsLower.second, lparamsUpper.second, pos2Dzy, dir2Dzy, covMat2Dzy, chi22Dzy);
    LogTrace("VectorHitBuilderAlgorithm") << "\t  pos2Dzy: " << pos2Dzy;
    LogTrace("VectorHitBuilderAlgorithm") << "\t  dir2Dzy: " << dir2Dzy;
    LogTrace("VectorHitBuilderAlgorithm") << "\t  cov2Dzy: " << covMat2Dzy;
    VectorHit2D vh2Dzy = VectorHit2D(pos2Dzy, dir2Dzy, covMat2Dzy, chi22Dzy);

    OmniClusterRef lowerOmni(lower); 
    OmniClusterRef upperOmni(upper); 
    VectorHit vh = VectorHit(*stack, vh2Dzx, vh2Dzy, lowerOmni, upperOmni);
    return vh;

  }

  return VectorHit();

}



void VectorHitBuilderAlgorithm::fit2Dzx(const Local3DPoint lpCI, const Local3DPoint lpCO,
                                        const LocalError leCI, const LocalError leCO,
                                        Local3DPoint& pos, Local3DVector& dir,
                                        AlgebraicSymMatrix22& covMatrix,
                                        double& chi2)
{
  std::vector<float> x = {lpCI.z(), lpCO.z()};
  std::vector<float> y = {lpCI.x(), lpCO.x()};
  float sqCI = sqrt(leCI.xx());
  float sqCO = sqrt(leCO.xx());
  std::vector<float> sigy = {sqCI, sqCO};

  fit(x,y,sigy,pos,dir,covMatrix,chi2);

  return;

}

void VectorHitBuilderAlgorithm::fit2Dzy(const Local3DPoint lpCI, const Local3DPoint lpCO,
                                        const LocalError leCI, const LocalError leCO,
                                        Local3DPoint& pos, Local3DVector& dir,
                                        AlgebraicSymMatrix22& covMatrix,
                                        double& chi2)
{
  std::vector<float> x = {lpCI.z(), lpCO.z()};
  std::vector<float> y = {lpCI.y(), lpCO.y()};
  float sqCI = sqrt(leCI.yy());
  float sqCO = sqrt(leCO.yy());
  std::vector<float> sigy = {sqCI, sqCO};

  fit(x,y,sigy,pos,dir,covMatrix,chi2);

  return;

}

void VectorHitBuilderAlgorithm::fit(const std::vector<float>& x,
                                    const std::vector<float>& y,
                                    const std::vector<float>& sigy,
                                    Local3DPoint& pos, Local3DVector& dir,
                                    AlgebraicSymMatrix22& covMatrix,
                                    double& chi2)
{

  if(x.size() != y.size() || x.size() != sigy.size()){
    edm::LogError("VectorHitBuilderAlgorithm") << "Different size for x,z !! No fit possible.";
    return;
  }

  float slope     = 0.;
  float intercept = 0.;
  float covss     = 0.;
  float covii     = 0.;
  float covsi     = 0.;

  theFitter->fit(x,y,x.size(),sigy,slope,intercept,covss,covii,covsi);

  covMatrix[0][0] = covss; // this is var(dy/dz)
  covMatrix[1][1] = covii; // this is var(y)
  covMatrix[1][0] = covsi; // this is cov(dy/dz,y)

  for (unsigned int j=0; j < x.size(); j++){
    const double ypred = intercept + slope*x[j];
    const double dy = (y[j] - ypred)/sigy[j];
    chi2 += dy*dy;
 }

  pos = Local3DPoint(intercept,0.,0.);
  if(x.size()==2){
    //difference in z is the difference of the lowermost and the uppermost cluster z pos
    float slopeZ = x.at(1) - x.at(0);
    dir = LocalVector(slope,0.,slopeZ);
  } else {
    dir = LocalVector(slope,0.,-1.);
  }

}

