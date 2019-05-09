#ifndef Geometry_MTDGeometryBuilder_MTDDigiGeometryESModule_H
#define Geometry_MTDGeometryBuilder_MTDDigiGeometryESModule_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include <memory>

#include <string>

namespace edm {
  class ConfigurationDescriptions;
}

class MTDTopology;
class PMTDParameters;
class Alignments;
class AlignmentErrorsExtended;
class AlignmentSurfaceDeformations;

class  MTDDigiGeometryESModule: public edm::ESProducer{
 public:
  MTDDigiGeometryESModule(const edm::ParameterSet & p);
  ~MTDDigiGeometryESModule() override; 
  std::unique_ptr<MTDGeometry> produce(const MTDDigiGeometryRecord &);

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  
 private:
  /// Called when geometry description changes
  const std::string alignmentsLabel_;
  const std::string myLabel_;
  edm::ESGetToken<GeometricTimingDet, IdealGeometryRecord> geomToken_;
  edm::ESGetToken<MTDTopology, MTDTopologyRcd> topoToken_;
  edm::ESGetToken<PMTDParameters, PMTDParametersRcd> paramToken_;
  edm::ESGetToken<Alignments, GlobalPositionRcd> globalPositionToken_;
  edm::ESGetToken<Alignments, MTDAlignmentRcd> alignmentsToken_;
  edm::ESGetToken<AlignmentErrorsExtended, MTDAlignmentErrorExtendedRcd> alignmentErrorsToken_;
  edm::ESGetToken<AlignmentSurfaceDeformations, MTDSurfaceDeformationRcd> surfaceDeformationsToken_;
  bool applyAlignment_; // Switch to apply alignment corrections
  bool fromDDD_;
};

#endif
