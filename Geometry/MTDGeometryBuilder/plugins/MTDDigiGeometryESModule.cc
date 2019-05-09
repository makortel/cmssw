#include "MTDDigiGeometryESModule.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeomBuilderFromGeometricTimingDet.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/PMTDParametersRcd.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "CondFormats/GeometryObjects/interface/PMTDParameters.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"

// Alignments
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CondFormats/Alignment/interface/AlignmentSurfaceDeformations.h"
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/AlignmentRecord/interface/MTDAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/MTDAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/MTDSurfaceDeformationRcd.h"
#include "Geometry/CommonTopologies/interface/GeometryAligner.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <memory>

//__________________________________________________________________
MTDDigiGeometryESModule::MTDDigiGeometryESModule(const edm::ParameterSet & p) 
  : alignmentsLabel_(p.getParameter<std::string>("alignmentsLabel")),
    myLabel_(p.getParameter<std::string>("appendToDataLabel"))
{
    applyAlignment_ = p.getParameter<bool>("applyAlignment");
    fromDDD_ = p.getParameter<bool>("fromDDD");

    auto cc = setWhatProduced(this);
    geomToken_ = cc.consumesFrom<GeometricTimingDet, IdealGeometryRecord>(edm::ESInputTag{});
    topoToken_ = cc.consumesFrom<MTDTopology, MTDTopologyRcd>(edm::ESInputTag{});
    paramToken_ = cc.consumesFrom<PMTDParameters, PMTDParametersRcd>(edm::ESInputTag{});
    if(applyAlignment_) {
      globalPositionToken_ = cc.consumesFrom<Alignments, GlobalPositionRcd>(edm::ESInputTag{"", alignmentsLabel_});
      alignmentsToken_ = cc.consumesFrom<Alignments, MTDAlignmentRcd>(edm::ESInputTag{"", alignmentsLabel_});
      alignmentErrorsToken_ = cc.consumesFrom<AlignmentErrorsExtended, MTDAlignmentErrorExtendedRcd>(edm::ESInputTag{"", alignmentsLabel_});
      surfaceDeformationsToken_ = cc.consumesFrom<AlignmentSurfaceDeformations, MTDSurfaceDeformationRcd>(edm::ESInputTag{"", alignmentsLabel_});
    }

    edm::LogInfo("Geometry") << "@SUB=MTDDigiGeometryESModule"
			     << "Label '" << myLabel_ << "' "
			     << (applyAlignment_ ? "looking for" : "IGNORING")
			     << " alignment labels '" << alignmentsLabel_ << "'.";
}

//__________________________________________________________________
MTDDigiGeometryESModule::~MTDDigiGeometryESModule() {}

void
MTDDigiGeometryESModule::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription descDB;
  descDB.add<std::string>( "appendToDataLabel", "" );
  descDB.add<bool>( "fromDDD", false );
  descDB.add<bool>( "applyAlignment", true );
  descDB.add<std::string>( "alignmentsLabel", "" );
  descriptions.add( "mtdGeometryDB", descDB );

  edm::ParameterSetDescription desc;
  desc.add<std::string>( "appendToDataLabel", "" );
  desc.add<bool>( "fromDDD", true );
  desc.add<bool>( "applyAlignment", true );
  desc.add<std::string>( "alignmentsLabel", "" );
  descriptions.add( "mtdGeometry", desc );
}

//__________________________________________________________________
std::unique_ptr<MTDGeometry>
MTDDigiGeometryESModule::produce(const MTDDigiGeometryRecord & iRecord)
{ 
  //
  // Called whenever the alignments, alignment errors or global positions change
  //
  const auto& gD = iRecord.get(geomToken_);
  const auto& tTopoR = iRecord.get(topoToken_);
  const MTDTopology *tTopo = &tTopoR;
  const auto& ptp = iRecord.get(paramToken_);
  
  MTDGeomBuilderFromGeometricTimingDet builder;
  std::unique_ptr<MTDGeometry> mtd(builder.build(&gD, ptp, tTopo));

  if (applyAlignment_) {
    // Since fake is fully working when checking for 'empty', we should get rid of applyAlignment_!
    const auto& globalPosition = iRecord.get(globalPositionToken_);
    const auto& alignments = iRecord.get(alignmentsToken_);
    const auto& alignmentErrors = iRecord.get(alignmentErrorsToken_);
    // apply if not empty:
    if (alignments.empty() && alignmentErrors.empty() && globalPosition.empty()) {
      edm::LogInfo("Config") << "@SUB=MTDDigiGeometryRecord::produce"
			     << "Alignment(Error)s and global position (label '"
	 		     << alignmentsLabel_ << "') empty: Geometry producer (label "
			     << "'" << myLabel_ << "') assumes fake and does not apply.";
    } else {
      GeometryAligner ali;
      ali.applyAlignments<MTDGeometry>(mtd.get(), &alignments, &alignmentErrors,
				       align::DetectorGlobalPosition(globalPosition,
								     DetId(DetId::Forward)));
    }

    const auto& surfaceDeformations = iRecord.get(surfaceDeformationsToken_);
    // apply if not empty:
    if (surfaceDeformations.empty()) {
      edm::LogInfo("Config") << "@SUB=MTDDigiGeometryRecord::produce"
			     << "AlignmentSurfaceDeformations (label '"
			     << alignmentsLabel_ << "') empty: Geometry producer (label "
			     << "'" << myLabel_ << "') assumes fake and does not apply.";
    } else {
      GeometryAligner ali;
      ali.attachSurfaceDeformations<MTDGeometry>(mtd.get(), &surfaceDeformations);
    }
  }
  
  
  return mtd;
}

DEFINE_FWK_EVENTSETUP_MODULE(MTDDigiGeometryESModule);
