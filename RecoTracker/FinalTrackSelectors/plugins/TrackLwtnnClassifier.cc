#include "RecoTracker/FinalTrackSelectors/interface/TrackMVAClassifier.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "getBestVertex.h"

//from lwtnn
#include "lwtnn/LightweightNeuralNetwork.hh"
#include "lwtnn/parse_json.hh"
#include <fstream>

namespace {
  struct lwtnn {
    lwtnn(const edm::ParameterSet& cfg){
      // TODO: investigate if the construction of the NN could be
      // moved to an ESProducer (or if it is worth of the effort) as
      // in our case we will use the same network N times with
      // different input collections (and cuts).

//      const auto fileName = cfg.getParameter<edm::FileInPath>("fileName");

//      std::ifstream jsonfile(fileName.fullPath().c_str());
      std::ifstream jsonfile("/afs/cern.ch/work/j/jhavukai/private/LWTNNinCMSSW/CMSSW_9_4_0_pre3/src/RecoTracker/FinalTrackSelectors/data/neural_net.json"); 
      auto config = lwt::parse_json(jsonfile);

      neuralNetwork_ = std::make_unique<lwt::LightweightNeuralNetwork>(config.inputs, config.layers, config.outputs);
    }

    static const char *name() { return "TrackLwtnnClassifier"; }

    static void fillDescriptions(edm::ParameterSetDescription& desc) {
//      desc.add<edm::FileInPath>("fileName", edm::FileInPath());
    }

    void beginStream() {}
    void initEvent(const edm::EventSetup& es) {}

    float operator()(reco::Track const & trk,
                     reco::BeamSpot const & beamSpot,
                     reco::VertexCollection const & vertices) {

      Point bestVertex = getBestVertex(trk,vertices);

      inputs_["trk_pt"] = trk.pt();
      inputs_["trk_eta"] = trk.eta();
      inputs_["trk_lambda"] = trk.lambda();
      inputs_["trk_dxy"] = trk.dxy(beamSpot.position()); // is the training with abs() or not?
      inputs_["trk_dz"] = trk.dz(beamSpot.position()); // is the training with abs() or not?
      inputs_["trk_dxyClosestPV"] = trk.dxy(bestVertex); // is the training with abs() or not?
      inputs_["trk_dzClosestPV"] = trk.dz(bestVertex); // is the training with abs() or not?
      inputs_["trk_ptErr"] = trk.ptError();
      inputs_["trk_etaErr"] = trk.etaError();
      inputs_["trk_lambdaErr"] = trk.lambdaError();
      inputs_["trk_dxyErr"] = trk.dxyError();
      inputs_["trk_dzErr"] = trk.dzError();
      inputs_["trk_nChi2"] = trk.normalizedChi2();
      inputs_["trk_ndof"] = trk.ndof();
      inputs_["trk_nInvalid"] = trk.hitPattern().numberOfLostHits(reco::HitPattern::TRACK_HITS);
      inputs_["trk_nPixel"] = trk.hitPattern().numberOfValidPixelHits();
      inputs_["trk_nStrip"] = trk.hitPattern().numberOfValidStripHits();
      inputs_["trk_nPixelLay"] = trk.hitPattern().pixelLayersWithMeasurement();
      inputs_["trk_nStripLay"] = trk.hitPattern().stripLayersWithMeasurement();
      inputs_["trk_n3DLay"] = (trk.hitPattern().numberOfValidStripLayersWithMonoAndStereo()+trk.hitPattern().pixelLayersWithMeasurement());
      inputs_["trk_nLostLay"] = trk.hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::TRACK_HITS);
      inputs_["trk_algo"] = trk.algo(); // eventually move to originalAlgo

      auto out = neuralNetwork_->compute(inputs_);
      // there should only one output
      if(out.size() != 1) throw cms::Exception("LogicError") << "Expecting exactly one output from NN, got " << out.size();


      float output = 2.0*out.begin()->second-1.0;
      return output;
    }


    std::unique_ptr<lwt::LightweightNeuralNetwork> neuralNetwork_;
    lwt::ValueMap inputs_; //typedef of map<string, double>
  };

  using TrackLwtnnClassifier = TrackMVAClassifier<lwtnn>;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TrackLwtnnClassifier);
