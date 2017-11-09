#ifndef RecoLocalTracker_Phase2TrackerRecHits_VectorHitBuilderAlgorithm_H
#define RecoLocalTracker_Phase2TrackerRecHits_VectorHitBuilderAlgorithm_H

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackGeomDet.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPE.h"
#include "DataFormats/TrackerRecHit2D/interface/VectorHit.h"
#include "CommonTools/Statistics/interface/LinearFit.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class VectorHitBuilderAlgorithm {
 public:
  typedef edm::Ref<edmNew::DetSetVector<Phase2TrackerCluster1D>, Phase2TrackerCluster1D> Phase2TrackerCluster1DRef;
  typedef edmNew::DetSet<Phase2TrackerCluster1D> detset;
  typedef detset::const_iterator const_iterator;
  typedef edmNew::DetSetVector<VectorHit> output_t;
  typedef std::pair< StackGeomDet, std::vector<Phase2TrackerCluster1D> > StackClusters;

  VectorHitBuilderAlgorithm(const edm::ParameterSet&);
  ~VectorHitBuilderAlgorithm() { delete theFitter; };
  void initialize(const edm::EventSetup&);
  void initTkGeom(edm::ESHandle< TrackerGeometry > tkGeomHandle);
  void initTkTopo(edm::ESHandle< TrackerTopology > tkTopoHandle);
  void initCpe(const ClusterParameterEstimator<Phase2TrackerCluster1D>* cpeProd);

  // compute parallax correction for window cut
  double computeParallaxCorrection(const PixelGeomDetUnit*&, const Point3DBase<float, LocalTag>&, const PixelGeomDetUnit*&, const Point3DBase<float, LocalTag>&);

  // debug
  void printClusters(const edmNew::DetSetVector<Phase2TrackerCluster1D>& clusters);
  void printCluster(const GeomDet* geomDetUnit, const Phase2TrackerCluster1D* cluster);

  void loadDetSetVector( std::map< DetId,std::vector<VectorHit> >& theMap, edmNew::DetSetVector<VectorHit>& theCollection ) const ;

  void run(edm::Handle< edmNew::DetSetVector<Phase2TrackerCluster1D> > clusters, 
           VectorHitCollectionNew& vhAcc, VectorHitCollectionNew& vhRej,
           edmNew::DetSetVector<Phase2TrackerCluster1D>& clustersAcc, edmNew::DetSetVector<Phase2TrackerCluster1D>& clustersRej );

  bool checkClustersCompatibilityBeforeBuilding(edm::Handle< edmNew::DetSetVector<Phase2TrackerCluster1D> > clusters,
                                                                         const detset & theLowerDetSet,
                                                                         const detset & theUpperDetSet);
  //not implemented yet
  bool checkClustersCompatibility(Local3DPoint& posinner, Local3DPoint& posouter, LocalError& errinner, LocalError& errouter);

  class LocalPositionSort {
    public: 
      LocalPositionSort(const TrackerGeometry *geometry, const ClusterParameterEstimator<Phase2TrackerCluster1D>* cpe, const GeomDet * geomDet) : 
        geom_(geometry), cpe_(cpe), geomDet_(geomDet) {}
      bool operator()(Phase2TrackerCluster1DRef clus1, Phase2TrackerCluster1DRef clus2) const ;
    private:
      const TrackerGeometry *geom_;
      const ClusterParameterEstimator<Phase2TrackerCluster1D>* cpe_;
      const GeomDet * geomDet_;
  };

  std::vector<std::pair<VectorHit,bool>> buildVectorHits(const StackGeomDet * stack, 
                                           edm::Handle< edmNew::DetSetVector<Phase2TrackerCluster1D> > clusters,
                                           const detset & DSVinner, const detset & DSVouter,
                                           const std::vector<bool>& phase2OTClustersToSkip = std::vector<bool>());

  VectorHit buildVectorHit(const StackGeomDet* stack, Phase2TrackerCluster1DRef lower, Phase2TrackerCluster1DRef upper);

  void fit2Dzx(const Local3DPoint lpCI, const Local3DPoint lpCO,
               const LocalError leCI, const LocalError leCO,
               Local3DPoint& pos, Local3DVector& dir,
               AlgebraicSymMatrix22& covMatrix, double& chi2);
  void fit2Dzy(const Local3DPoint lpCI, const Local3DPoint lpCO,
               const LocalError leCI, const LocalError leCO,
               Local3DPoint& pos, Local3DVector& dir,
               AlgebraicSymMatrix22& covMatrix, double& chi2);

  void fit(const std::vector<float>& x,
           const std::vector<float>& y,
           const std::vector<float>& sigy,
           Local3DPoint& pos, Local3DVector& dir,
           AlgebraicSymMatrix22& covMatrix, double& chi2);


 private:
  edm::ESInputTag cpeTag_;
  const TrackerGeometry* theTkGeom;
  const TrackerTopology* theTkTopo;
  const ClusterParameterEstimator<Phase2TrackerCluster1D>* cpe;
  unsigned int nMaxVHforeachStack;
  std::vector< double > barrelCut;
  std::vector< double > endcapCut;
  LinearFit* theFitter;

};

#endif
