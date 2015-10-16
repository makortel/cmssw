#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/VecArray.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/libminifloat.h"
#include "DataFormats/PatCandidates/interface/liblogintpack.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include <iomanip>

namespace {
  float packUnpack(float value) {
    return MiniFloatConverter::float16to32(MiniFloatConverter::float32to16(value));
  }

  class HitPatternPrinter {
  public:
    explicit HitPatternPrinter(const reco::Track& trk): track(trk) {}

    void print(std::ostream& os) const {
      const reco::HitPattern &p = track.hitPattern();

      for (int i = 0; i < p.numberOfHits(reco::HitPattern::TRACK_HITS); ++i) {
        uint32_t hit = p.getHitPattern(reco::HitPattern::TRACK_HITS, i);

        detLayer(os, p, hit);
        if(p.missingHitFilter(hit)) {
          os << "(miss)";
        }
        else if(p.inactiveHitFilter(hit)) {
          os << "(inact)";
        }
        else if(p.badHitFilter(hit)) {
          os << "(bad)";
        }
        os << " ";
      }

      if(p.numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS) > 0) {
        os << "lost inner ";

        for (int i = 0; i < p.numberOfHits(reco::HitPattern::MISSING_INNER_HITS); ++i) {
          uint32_t hit = p.getHitPattern(reco::HitPattern::MISSING_INNER_HITS, i);

          if(p.missingHitFilter(hit)) {
            detLayer(os, p, hit);
            os << " ";
          }
        }
      }
      if(p.numberOfLostHits(reco::HitPattern::MISSING_OUTER_HITS) > 0) {
        os << "lost outer ";

        for (int i = 0; i < p.numberOfHits(reco::HitPattern::MISSING_OUTER_HITS); ++i) {
          uint32_t hit = p.getHitPattern(reco::HitPattern::MISSING_OUTER_HITS, i);

          if(p.missingHitFilter(hit)) {
            detLayer(os, p, hit);
            os << " ";
          }
        }
      }
    }

  private:
    static void detLayer(std::ostream& os, const reco::HitPattern& p, uint32_t hit) {
      if(p.pixelBarrelHitFilter(hit)) {
        os << "BPIX";
      }
      else if(p.pixelEndcapHitFilter(hit)) {
        os << "FPIX";
      }
      else if(p.stripTIBHitFilter(hit)) {
        os << "TIB";
      }
      else if(p.stripTIDHitFilter(hit)) {
        os << "TID";
      }
      else if(p.stripTOBHitFilter(hit)) {
        os << "TOB";
      }
      else if(p.stripTECHitFilter(hit)) {
        os << "TEC";
      }
      os << p.getLayer(hit);
    }

    const reco::Track& track;
  };

  std::ostream& operator<<(std::ostream& os, const HitPatternPrinter& hpp) {
    hpp.print(os);
    return os;
  }

  class TrackAlgoPrinter {
  public:
    explicit TrackAlgoPrinter(const reco::Track& trk): track(trk) {}

    void print(std::ostream& os) const {
      edm::VecArray<reco::TrackBase::TrackAlgorithm, reco::TrackBase::algoSize> algos;
      for(int ialgo=0; ialgo < reco::TrackBase::algoSize; ++ialgo) {
        auto algo = static_cast<reco::TrackBase::TrackAlgorithm>(ialgo);
        if(track.isAlgoInMask(algo)) {
          algos.push_back(algo);
        }
      }

      os << "algo " << reco::TrackBase::algoName(track.algo());
      if(track.originalAlgo() != track.algo())
        os << " originalAlgo " << reco::TrackBase::algoName(track.originalAlgo());
      if(algos.size() > 1) {
        os << " algoMask";
        for(auto algo: algos) {
          os << " " << reco::TrackBase::algoName(algo);
        }
      }
    }

  private:
    const reco::Track& track;
  };
  std::ostream& operator<<(std::ostream& os, const TrackAlgoPrinter& tap) {
    tap.print(os);
    return os;
  }

  class DzCalculationPrinter {
  public:
    using Point = pat::PackedCandidate::Point;
    using Vector = pat::PackedCandidate::Vector;

    DzCalculationPrinter(const Point& ref_, const Point& point_, const Vector& momentum_, double phi_):
      ref(ref_), point(point_), momentum(momentum_), phi(phi_) {}

    void print(std::ostream& os) const {
      printPhi(os);
      os << " ";
      printDot(os);
    }

    void printPhi(std::ostream& os) const {
      os << "phi:";
      const auto diffx = ref.X()-point.X();
      const auto diffy = ref.Y()-point.Y();
      const auto diffz = ref.Z()-point.Z();
      os << " diff x " << diffx << " y " << diffy << " z " << diffz;

      const auto cosphi = std::cos(phi);
      const auto sinphi = std::sin(phi);
      os << " phi " << phi << " cosphi " << cosphi << " sinphi " << sinphi;

      const auto xterm = diffx*cosphi;
      const auto yterm = diffy*sinphi;
      os << " xterm " << xterm << " yterm " << yterm;

      const auto pzpt = momentum.z()/std::sqrt(momentum.perp2());
      os << " pzpt " << pzpt;

      const auto secondterm = (xterm+yterm)*pzpt;
      os << " secondterm " << secondterm;

      const auto result = diffz - secondterm;
      os << " result " << result;
    }

    void printDot(std::ostream& os) const {
      os << "dot:";
      const auto diffx = ref.X()-point.X();
      const auto diffy = ref.Y()-point.Y();
      const auto diffz = ref.Z()-point.Z();
      //os << " diff x " << diffx << " y " << diffy << " z " << diffz;

      const auto scalex = momentum.x()/std::sqrt(momentum.perp2());
      const auto scaley = momentum.y()/std::sqrt(momentum.perp2());
      os << " scalex " << scalex << " scaley " << scaley;

      const auto xterm = diffx*scalex;
      const auto yterm = diffy*scaley;
      os << " xterm " << xterm << " yterm " << yterm;

      const auto pzpt = momentum.z()/std::sqrt(momentum.perp2());
      os << " pzpt " << pzpt;

      const auto secondterm = (xterm+yterm)*pzpt;
      os << " secondterm " << secondterm;

      const auto result = diffz - secondterm;
      os << " result " << result;
    }

  private:
    const Point ref;
    const Point point;
    const Vector momentum;
    const double phi;
  };
  std::ostream& operator<<(std::ostream& os, const DzCalculationPrinter& dcp) {
    dcp.print(os);
    return os;
  }

  class DxyCalculationPrinter {
  public:
    using Point = pat::PackedCandidate::Point;
    using Vector = pat::PackedCandidate::Vector;

    DxyCalculationPrinter(const Point& ref_, const Point& point_, const Vector& momentum_, double phi_):
      ref(ref_), point(point_), momentum(momentum_), phi(phi_) {}

    void print(std::ostream& os) const {
      printPhi(os);
      os << " ";
      printDot(os);
    }

    void printPhi(std::ostream& os) const {
      os << "phi:";
      const auto diffx = ref.X()-point.X();
      const auto diffy = ref.Y()-point.Y();
      os << " diff x " << diffx << " y " << diffy;

      const auto sinphi = std::sin(phi);
      const auto cosphi = std::cos(phi);
      os << " phi " << phi << " sinphi " << cosphi << " cosphi " << sinphi;

      const auto xterm = diffx*sinphi;
      const auto yterm = diffy*cosphi;
      os << " xterm " << xterm << " yterm " << yterm;

      const auto result = -xterm + yterm;
      os << " result " << result;
    }

    void printDot(std::ostream& os) const {
      os << "dot:";
      const auto diffx = ref.X()-point.X();
      const auto diffy = ref.Y()-point.Y();
      os << " diff x " << diffx << " y " << diffy;

      const auto scalex = momentum.y()/std::sqrt(momentum.perp2());
      const auto scaley = momentum.x()/std::sqrt(momentum.perp2());
      os << " scalex " << scalex << " scaley " << scaley;

      const auto xterm = diffx*scalex;
      const auto yterm = diffy*scaley;
      os << " xterm " << xterm << " yterm " << yterm;

      const auto result = -xterm + yterm;
      os << " result " << result;
    }

  private:
    const Point ref;
    const Point point;
    const Vector momentum;
    const double phi;
  };
  std::ostream& operator<<(std::ostream& os, const DxyCalculationPrinter& dcp) {
    dcp.print(os);
    return os;
  }

  class CovPrinter {
  public:
    CovPrinter(const reco::Track& trackPc_, const reco::Track& track_, int i_, int j_, double multip_=1.0):
      trackPc(trackPc_), track(track_), i(i_), j(j_), multip(multip_) {}

    void print(std::ostream& os) const {
      const auto repacked = multip != 1.0 ? packUnpack(track.covariance(i, j)*multip)/multip : packUnpack(track.covariance(i, j));
      os << trackPc.covariance(i, j) << " " << track.covariance(i, j) << " (" << repacked << ")";
    }

  private:
    const reco::Track& trackPc;
    const reco::Track& track;
    const int i;
    const int j;
    const double multip;
  };
  std::ostream& operator<<(std::ostream& os, const CovPrinter& cp) {
    cp.print(os);
    return os;
  }

  double diffRelative(double a, double b) {
    return (a-b)/b;
  }

  class LogIntHelper {
  public:
    LogIntHelper(double lmin, double lmax): lmin_(lmin), lmax_(lmax) {}

    enum {
      stat_OK = 0,
      stat_under_OK = 1,
      stat_under_notOK = 2,
      stat_over_OK = 3,
      stat_over_notOK = 4
    };

    void book(DQMStore::IBooker& iBooker,
              const std::string& name, const std::string& title,
              int nbins,double min,double max,
              int flow_nbins,double flow_min,double flow_max) {
      hInrange = iBooker.book1D(name, title, nbins, min, max);
      hUnderOverflow = iBooker.book1D(name+"UnderOverlFlow", title+" with over- and underflow", flow_nbins, flow_min, flow_max);
      hStatus = iBooker.book1D(name+"Status", title+" status", 5, -0.5, 4.5);
      hStatus->setBinLabel(1, "OK");
      hStatus->setBinLabel(2, "Underflow, PC is min");
      hStatus->setBinLabel(3, "Underflow, PC is not min");
      hStatus->setBinLabel(4, "Overflow, PC is max");
      hStatus->setBinLabel(5, "Overflow, PC is not max");
    }

    double largestValue() const {
      return logintpack::unpack8log(127, lmin_, lmax_);
    }

    double smallestNonZeroValue() const {
      return logintpack::unpack8log(0, lmin_, lmax_);
    }

    enum class RangeStatus {
      underflow,
      inrange,
      overflow
    };

    class Result {
    public:
      Result(RangeStatus status, double diff, double pcvalue, double trackvalue):
        diff_(diff), pcvalue_(pcvalue), trackvalue_(trackvalue), status_(status)
      {}

      RangeStatus status() const { return status_; }
      double diff() const { return diff_; }

      bool outsideExpectedRange(double min, double max) const {
        if(status_ != RangeStatus::inrange)
          return false;
        return diff_ < min || diff_ > max;
      }

      void print(std::ostream& os) const {
        os << diff_ << " ";
        if(status_ == RangeStatus::underflow)
          os << " (underflow) ";
        else if(status_ == RangeStatus::overflow)
          os << " (overflow) ";
        os << pcvalue_ << " " << trackvalue_;
      }

    private:
      const double diff_;
      const double pcvalue_;
      const double trackvalue_;
      const RangeStatus status_;
    };

    Result fill(double pcvalue, double trackvalue) {
      const auto diff = diffRelative(pcvalue, trackvalue);

      const auto tmp = std::abs(trackvalue);
      RangeStatus status;
      if(tmp > largestValue()) {
        status = RangeStatus::overflow;
        hUnderOverflow->Fill(diff);
        if(pcvalue == largestValue()) {
          hStatus->Fill(stat_over_OK);
        }
        else {
          hStatus->Fill(stat_over_notOK);
        }
      }
      else if(tmp < smallestNonZeroValue()) {
        status = RangeStatus::underflow;
        hUnderOverflow->Fill(diff);
        if(pcvalue == smallestNonZeroValue()) {
          hStatus->Fill(stat_under_OK);
        }
        else {
          hStatus->Fill(stat_under_notOK);
        }
      }
      else {
        status = RangeStatus::inrange;
        hInrange->Fill(diff);
        hStatus->Fill(stat_OK);
      }

      return Result(status, diff, pcvalue, trackvalue);
    }

  private:
    const double lmin_;
    const double lmax_;

    MonitorElement *hInrange;
    MonitorElement *hUnderOverflow;
    MonitorElement *hStatus;
  };
  std::ostream& operator<<(std::ostream& os, const LogIntHelper::Result& res) {
    res.print(os);
    return os;
  }
}

class PackedCandidateTrackValidator: public DQMEDAnalyzer{
 public:
  PackedCandidateTrackValidator(const edm::ParameterSet& pset);
  virtual ~PackedCandidateTrackValidator();

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup& ) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:

  edm::EDGetTokenT<edm::View<reco::Track>> tracksToken_;
  edm::EDGetTokenT<reco::VertexCollection> verticesToken_;
  edm::EDGetTokenT<reco::VertexCollection> slimmedVerticesToken_;
  edm::EDGetTokenT<edm::Association<pat::PackedCandidateCollection>> trackToPackedCandidateToken_;

  std::string rootFolder_;

  enum {
    sf_AllTracks = 0,
    sf_AssociatedToPC = 1,
    sf_PCIsCharged = 2,
    sf_PCHasTrack = 3,
    sf_PCIsNotElectron = 4,
    sf_PCHasHits = 5,
    sf_PCNdofNot0 = 6,
    sf_NoMissingInnerHits = 7
  };
  MonitorElement *h_selectionFlow;

  MonitorElement *h_diffPx;
  MonitorElement *h_diffPy;
  MonitorElement *h_diffPz;

  MonitorElement *h_diffVx;
  MonitorElement *h_diffVy;
  MonitorElement *h_diffVz;
  /*
  MonitorElement *h_diffVxVsVertex;
  MonitorElement *h_diffVyVsVertex;
  MonitorElement *h_diffVzVsVertex;
  */

  MonitorElement *h_diffNormalizedChi2;
  MonitorElement *h_diffNdof;

  MonitorElement *h_diffCharge;
  MonitorElement *h_diffIsHighPurity;

  MonitorElement *h_diffQoverp;
  MonitorElement *h_diffPt;
  MonitorElement *h_diffEta;
  MonitorElement *h_diffTheta;
  MonitorElement *h_diffPhi;
  MonitorElement *h_diffDxyAssocPV;
  MonitorElement *h_diffDxyPV;
  MonitorElement *h_diffDzAssocPV;
  MonitorElement *h_diffDzPV;

  MonitorElement *h_diffTrackDxy;
  MonitorElement *h_diffTrackDz;

  //MonitorElement *h_diffQoverpError;
  LogIntHelper h_diffQoverpError;
  MonitorElement *h_diffPtError;
  MonitorElement *h_diffEtaError;
  MonitorElement *h_diffThetaError;
  MonitorElement *h_diffPhiError;
  MonitorElement *h_diffDxyError;
  MonitorElement *h_diffDzError;
  MonitorElement *h_diffDszError;
  MonitorElement *h_diffDzErrorInf;

  MonitorElement *h_diffTrackDxyError;
  MonitorElement *h_diffTrackDzError;


  /*
  MonitorElement *h_diffCovQoverpLambda;
  MonitorElement *h_diffCovQoverpPhi;
  MonitorElement *h_diffCovQoverpDxy;
  MonitorElement *h_diffCovQoverpDz;
  MonitorElement *h_diffCovLambdaPhi;
  MonitorElement *h_diffCovLambdaDxy;
  */
  MonitorElement *h_diffCovLambdaDz;
  MonitorElement *h_diffCovPhiDxy;
  MonitorElement *h_diffCovPhiDz;
  MonitorElement *h_diffCovDxyDz;

  MonitorElement *h_diffNumberOfPixelHits;
  MonitorElement *h_diffNumberOfHits;
  MonitorElement *h_diffLostInnerHits;

  MonitorElement *h_diffHitPatternNumberOfValidPixelHits;
  MonitorElement *h_diffHitPatternNumberOfValidHits;
  MonitorElement *h_diffHitPatternNumberOfLostInnerHits;
  MonitorElement *h_diffHitPatternHasValidHitInFirstPixelBarrel;

  MonitorElement *h_numberPixelHitsOverMax;
  MonitorElement *h_numberStripHitsOverMax;
  MonitorElement *h_numberHitsOverMax;
};

PackedCandidateTrackValidator::PackedCandidateTrackValidator(const edm::ParameterSet& iConfig):
  tracksToken_(consumes<edm::View<reco::Track>>(iConfig.getUntrackedParameter<edm::InputTag>("tracks"))),
  verticesToken_(consumes<reco::VertexCollection>(iConfig.getUntrackedParameter<edm::InputTag>("vertices"))),
  slimmedVerticesToken_(consumes<reco::VertexCollection>(iConfig.getUntrackedParameter<edm::InputTag>("slimmedVertices"))),
  trackToPackedCandidateToken_(consumes<edm::Association<pat::PackedCandidateCollection>>(iConfig.getUntrackedParameter<edm::InputTag>("trackToPackedCandidateAssociation"))),
  rootFolder_(iConfig.getUntrackedParameter<std::string>("rootFolder")),
  h_diffQoverpError(-15, 0)
{}

PackedCandidateTrackValidator::~PackedCandidateTrackValidator() {}

void PackedCandidateTrackValidator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.addUntracked<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
  desc.addUntracked<edm::InputTag>("vertices", edm::InputTag("offlinePrimaryVertices"));
  desc.addUntracked<edm::InputTag>("slimmedVertices", edm::InputTag("offlineSlimmedPrimaryVertices"));
  desc.addUntracked<edm::InputTag>("trackToPackedCandidateAssociation", edm::InputTag("packedPFCandidates"));
  desc.addUntracked<std::string>("rootFolder", "Tracking/PackedCandidate");

  descriptions.add("packedCandidateTrackValidator", desc);
}

void PackedCandidateTrackValidator::bookHistograms(DQMStore::IBooker& iBooker, edm::Run const&, edm::EventSetup const&) {
  iBooker.setCurrentFolder(rootFolder_);

  h_selectionFlow = iBooker.book1D("selectionFlow", "Track selection flow", 6, -0.5, 7.5);
  h_selectionFlow->setBinLabel(1, "All tracks");
  h_selectionFlow->setBinLabel(2, "Associated to PackedCandidate");
  h_selectionFlow->setBinLabel(3, "PC is charged"),
  h_selectionFlow->setBinLabel(4, "PC has track");
  h_selectionFlow->setBinLabel(5, "PC is not electron");
  h_selectionFlow->setBinLabel(6, "PC has hits");
  h_selectionFlow->setBinLabel(7, "PC ndof != 0");
  h_selectionFlow->setBinLabel(8, "Track: no missing inner hits");

  constexpr int diffBins = 50;
  //constexpr float diff = 1e-3;
  //constexpr float diffP = 1e-1;
  constexpr float diffRel = 5e-2;

  /*
  h_diffPx = iBooker.book1D("diffPx", "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in px()", diffBins, -1.1, 0.2); // not equal, drop?
  h_diffPy = iBooker.book1D("diffPy", "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in py()", diffBins, -1.1, 0.2); // not equal, drop?
  h_diffPz = iBooker.book1D("diffPz", "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in pz()", diffBins, -1.1, 0.2); // not equal, drop?
  */

  h_diffVx = iBooker.book1D("diffVx", "PackedCandidate::bestTrack() - reco::Track in vx()", diffBins, -0.2, 0.2); // not equal, keep?
  h_diffVy = iBooker.book1D("diffVy", "PackedCandidate::bestTrack() - reco::Track in vy()", diffBins, -0.2, 0.2); // not equal, keep?
  h_diffVz = iBooker.book1D("diffVz", "PackedCandidate::bestTrack() - reco::Track in vz()", diffBins, -0.4, 0.4); // not equal, keep?

  /*
  h_diffVxVsVertex = iBooker.book1D("diffVxVsVertex", "PackedCandidate::bestTrack()::vx() - reco::Vertex::x()", diffBins, -0.1, 0.05);
  h_diffVyVsVertex = iBooker.book1D("diffVyVsVertex", "PackedCandidate::bestTrack()::vy() - reco::Vertex::y()", diffBins, -0.1, 0.05);
  h_diffVzVsVertex = iBooker.book1D("diffVzVsVertex", "PackedCandidate::bestTrack()::vz() - reco::Vertex::z()", diffBins, -0.2, 0.1);
  */

  h_diffNormalizedChi2 = iBooker.book1D("diffNormalizedChi2", "PackedCandidate::bestTrack() - reco::Track in normalizedChi2()", 30, -1.5, 1.5); // expected difference in -1...0
  h_diffNdof = iBooker.book1D("diffNdof", "PackedCandidate::bestTrack() - reco::Track in ndof()", 33, -30.5, 2.5); // to monitor the difference

  h_diffCharge = iBooker.book1D("diffCharge", "PackedCandidate::bestTrack() - reco::Track in charge()", 5, -2.5, 2.5); // expect equality
  h_diffIsHighPurity = iBooker.book1D("diffIsHighPurity", "PackedCandidate::bestTrack() - reco::Track in quality(highPurity)", 3, -1.5, 1.5); // expect equality

  h_diffQoverp = iBooker.book1D("diffQoverp", "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in qoverp()", diffBins, -1.1, 1.1); // not equal, drop?
  h_diffPt     = iBooker.book1D("diffPt",     "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in pt()",     diffBins, -1.1, 1.1); // not equal, keep
  h_diffEta    = iBooker.book1D("diffEta",    "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in eta()",    diffBins, -0.2, 0.2); // not equal, keep
  h_diffTheta  = iBooker.book1D("diffTheta",  "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in theta()",  diffBins, -0.2, 0.2); // not equal, drop
  h_diffPhi    = iBooker.book1D("diffPhi",    "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in phi()",    diffBins, -0.1, 0.02); // expect equality within precision

  h_diffDxyAssocPV = iBooker.book1D("diffDxyAssocPV", "(PackedCandidate::dxy() - reco::Track::dxy(assocPV))/reco::Track",           diffBins, -0.002, 0.002); // expect equality within precision
  h_diffDxyPV      = iBooker.book1D("diffDxyPV",      "(PackedCandidate::dxy(PV) - reco::Track::dxy(PV))/reco::Track",              diffBins, -0.002, 0.002); // expect equality within precision
  h_diffDzAssocPV  = iBooker.book1D("diffDzAssocPV",  "(PackedCandidate::dzAssociatedPV() - reco::Track::dz(assocPV))/reco::Track", diffBins, -0.002, 0.002); // expect equality within precision
  h_diffDzPV       = iBooker.book1D("diffDzPV",       "(PackedCandidate::dz(PV) - reco::Track::dz(PV))/reco::Track",                diffBins, -0.002, 0.002); // expect equality wihtin precision
  h_diffTrackDxy   = iBooker.book1D("diffTrackDxy",   "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in dxy()",          diffBins, -0.2, diffRel); // not equal
  h_diffTrackDz    = iBooker.book1D("diffTrackDz",    "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in dz()",           diffBins, -0.2, diffRel); // not equal

  //h_diffQoverpError = iBooker.book1D("diffQoverpError", "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in qoverpError()", 40, -0.05, 0.15); // expect equality within precision (worst precision is exp(1/128*15) =~ 12 %
  h_diffQoverpError.book(iBooker, "diffQoverpError", "/PackedCandidate::bestTrack() - reco::Track)/reco::track in qoverpError()",
                         40, -0.05, 0.15,
                         50, -0.5, 0.5);

  h_diffPtError     = iBooker.book1D("diffPtError",     "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in ptError()",     diffBins, -1.1, 1.1); // not equal
  h_diffEtaError    = iBooker.book1D("diffEtaError",    "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in etaError()",    60, -0.15, 0.15); // not equal
  h_diffThetaError  = iBooker.book1D("diffThetaError",  "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in thetaError()",  40, -0.05, 0.15); // expect equality within precision worst precision is exp(1/128*(20-5)) =~ 12 % (multiplied by pt^2 in packing & unpacking)
  h_diffPhiError    = iBooker.book1D("diffPhiError",    "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in phiError()",    40, -0.05, 0.15); // expect equality within precision worst precision is exp(1/128*(20-5)) =~ 12 % (multiplied by pt^2 in packing & unpacking)
  h_diffDxyError    = iBooker.book1D("diffDxyError",    "(PackedCandidate::dxyError() - reco::Track::dxyError())/reco::Track",       diffBins, -0.001, 0.001); // expect equality within precision
  h_diffDzError     = iBooker.book1D("diffDzError",     "(PackedCandidate::dzError() - reco::Track::dzError())/reco::Track",         diffBins, -0.001, 0.001); // expect equality within precision (not currently the case)
  h_diffDszError    = iBooker.book1D("diffDszError",    "(PackedCandidate::dzError() - reco::Track::dszError())/reco::Track",        diffBins, -0.001, 0.001); // ideally, not equal, but for now they are
  //h_diffDzError     = iBooker.book1D("diffDzError",     "(PackedCandidate::dzError() - reco::Track::dzError())/reco::Track",     diffBins, -0.02, 0.002);
  h_diffDzErrorInf  = iBooker.book1D("diffDzErrorInf",  "isinf(PackedCandidate::dzError()) - wouldbeinf(reco::Track::dzError())",     3, -1.5, 1.5);

  h_diffTrackDxyError    = iBooker.book1D("diffTrackDxyError",    "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in dxyError()",    diffBins, -0.001, 0.001); // expect equality within precision (but drop)
  h_diffTrackDzError     = iBooker.book1D("diffTrackDzError",     "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in dzError()",     diffBins, -0.02, 0.02); // not equal, drop

  /*
  h_diffCovQoverpLambda = iBooker.book1D("diffCovQoverpLambda", "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in cov(qoverp, lambda)", 10, -1.1, -0.9); // expect to be 0, drop?
  h_diffCovQoverpPhi    = iBooker.book1D("diffCovQoverpPhi",    "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in cov(qoverp, phi)",    10, -1.1, -0.9); // expect to be 0, drop?
  h_diffCovQoverpDxy    = iBooker.book1D("diffCovQoverpDxy",    "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in cov(qoverp, dxy)",    10, -1.1, -0.9); // expect to be 0, drop?
  h_diffCovQoverpDz     = iBooker.book1D("diffCovQoverpDz",     "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in cov(qoverp, dz)",     10, -1.1, -0.9); // expect to be 0, drop?
  h_diffCovLambdaPhi    = iBooker.book1D("diffCovLambdaPhi",    "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in cov(lambda, phi)",    10, -1.1, -0.9); // expect to be 0, drop?
  h_diffCovLambdaDxy    = iBooker.book1D("diffCovLambdaDxy",    "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in cov(lambda, dxy)",    10, -1.1, -0.9); // expect to be 0, drop?
  */
  h_diffCovLambdaDz     = iBooker.book1D("diffCovLambdaDz",     "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in cov(lambda, dz)",     210, -4, 0.2); // expect equality within precision, worst precision is exp(1/128*(17-4) =~ 11 %
  h_diffCovPhiDxy       = iBooker.book1D("diffCovPhiDxy",       "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in cov(phi, dxy)",       diffBins, -1, 0.05); // expect equality within precision, wors precision is exp(1/128)*(17-4) =~ 11 %
  //h_diffCovPhiDz        = iBooker.book1D("diffCovPhiDz",        "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in cov(phi, dz)",        10, -1.1, -0.9); // expect to be 0, drop?
  h_diffCovDxyDz        = iBooker.book1D("diffCovDxyDz",        "(PackedCandidate::bestTrack() - reco::Track)/reco::Track in cov(dxy, dz)",        diffBins, -0.1, 0.1); // expect equality within precision

  h_diffNumberOfPixelHits = iBooker.book1D("diffNumberOfPixelHits", "PackedCandidate::numberOfPixelHits() - reco::Track::hitPattern::numberOfValidPixelHits()", 5, -2.5, 2.5); // expect equality 
  h_diffNumberOfHits      = iBooker.book1D("diffNumberOfHits",      "PackedCandidate::numberHits() - reco::Track::hitPattern::numberOfValidHits()",             5, -2.5, 2.5); // expect equality
  h_diffLostInnerHits     = iBooker.book1D("diffLostInnerHits",     "PackedCandidate::lostInnerHits() - reco::Track::hitPattern::numberOfLostHits(MISSING_INNER_HITS)",      5, -2.5, 2.5); // expect equality

  h_diffHitPatternNumberOfValidPixelHits = iBooker.book1D("diffHitPatternNumberOfValidPixelHits", "PackedCandidate::bestTrack() - reco::Track in hitPattern::numberOfValidPixelHits()",   13, -10.5, 2.5); // not equal
  h_diffHitPatternNumberOfValidHits      = iBooker.book1D("diffHitPatternNumberOfValidHits",      "PackedCandidate::bestTrack() - reco::Track in hitPattern::numberOfValidHits()",      13, -10.5, 2.5); // not equal
  h_diffHitPatternNumberOfLostInnerHits  = iBooker.book1D("diffHitPatternNumberOfLostPixelHits",  "PackedCandidate::bestTrack() - reco::Track in hitPattern::numberOfLostHits(MISSING_INNER_HITS)", 13, -10.5, 2.5); // not equal
  h_diffHitPatternHasValidHitInFirstPixelBarrel = iBooker.book1D("diffHitPatternHasValidHitInFirstPixelBarrel", "PackedCandidate::bestTrack() - reco::Track in hitPattern::hasValidHitInFirstPixelBarrel", 3, -1.5, 1.5); // expect equality

  h_numberPixelHitsOverMax = iBooker.book1D("numberPixelHitsOverMax", "Number of pixel hits over the maximum of PackedCandidate", 10, 0, 10);
  h_numberStripHitsOverMax = iBooker.book1D("numberStripHitsOverMax", "Number of strip hits over the maximum of PackedCandidate", 10, 0, 10);
  h_numberHitsOverMax = iBooker.book1D("numberHitsOverMax", "Number of hits over the maximum of PackedCandidate", 20, 0, 20);
}

namespace {
  template<typename T> void fillNoFlow(MonitorElement* h, T val){
    h->Fill(std::min(std::max(val,((T) h->getTH1()->GetXaxis()->GetXmin())),((T) h->getTH1()->GetXaxis()->GetXmax())));
  }

  bool isDzErrorPackedFinite(const reco::Track& track) {
    // dz*x needs to be in double, as it is such in PackedCandidate::packVtx;
    return float(track.covariance(reco::TrackBase::i_dsz, reco::TrackBase::i_dsz)*10000.) <= MiniFloatConverter::max32ConvertibleToMax16();
  }
}

void PackedCandidateTrackValidator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<edm::View<reco::Track>> htracks;
  iEvent.getByToken(tracksToken_, htracks);
  const auto& tracks = *htracks;

  edm::Handle<reco::VertexCollection> hvertices;
  iEvent.getByToken(verticesToken_, hvertices);
  const auto& vertices = *hvertices;

  edm::Handle<reco::VertexCollection> hslimmedvertices;
  iEvent.getByToken(slimmedVerticesToken_, hslimmedvertices);
  const auto& slimmedVertices = *hslimmedvertices;

  if(vertices.empty())
    return;
  const reco::Vertex& pv = vertices[0];
  const reco::Vertex& slimmedPV = slimmedVertices[0];

  edm::Handle<edm::Association<pat::PackedCandidateCollection>> hassoc;
  iEvent.getByToken(trackToPackedCandidateToken_, hassoc);
  const auto& trackToPackedCandidate = *hassoc;

  for(size_t i=0; i<tracks.size(); ++i) {
    auto trackPtr = tracks.ptrAt(i);
    const reco::Track& track = *trackPtr;
    h_selectionFlow->Fill(sf_AllTracks);

    pat::PackedCandidateRef pcRef = trackToPackedCandidate[trackPtr];
    if(pcRef.isNull()) {
      continue;
    }
    h_selectionFlow->Fill(sf_AssociatedToPC);

    // Filter out neutral PackedCandidates, some of them may have track associated, and for those the charge comparison fails
    if(pcRef->charge() == 0) {
      continue;
    }
    h_selectionFlow->Fill(sf_PCIsCharged);

    const reco::Track *trackPcPtr = pcRef->bestTrack();
    if(!trackPcPtr) {
      continue;
    }
    h_selectionFlow->Fill(sf_PCHasTrack);

    // Filter out electrons to avoid comparisons to PackedCandidates with GsfTrack
    if(std::abs(pcRef->pdgId()) == 11) {
      continue;
    }
    h_selectionFlow->Fill(sf_PCIsNotElectron);

    // Filter out PackedCandidate-tracks with no hits, as they won't have their details filled
    const reco::Track& trackPc = *trackPcPtr;
    if(trackPc.hitPattern().numberOfValidHits() == 0) {
      continue;
    }
    h_selectionFlow->Fill(sf_PCHasHits);

    auto slimmedVertexRef = pcRef->vertexRef();
    const reco::Vertex& pcVertex = vertices[slimmedVertexRef.key()];

    /*
    fillNoFlow(h_diffPx, diffRelative(trackPc.px(), track.px()));
    fillNoFlow(h_diffPy, diffRelative(trackPc.py(), track.py()));
    fillNoFlow(h_diffPz, diffRelative(trackPc.pz(), track.pz()));
    */

    fillNoFlow(h_diffVx, trackPc.vx() - track.vx());
    fillNoFlow(h_diffVy, trackPc.vy() - track.vy());
    fillNoFlow(h_diffVz, trackPc.vz() - track.vz());

    /*
    fillNoFlow(h_diffVxVsVertex, trackPc.vx() - pcVertex.x());
    fillNoFlow(h_diffVyVsVertex, trackPc.vy() - pcVertex.y());
    fillNoFlow(h_diffVzVsVertex, trackPc.vz() - pcVertex.z());
    */

    // PackedCandidate recalculates the ndof in unpacking as
    // (nhits+npixelhits-5), but some strip hits may have dimension 2.
    // If PackedCandidate has ndof=0, the resulting normalizedChi2
    // will be 0 too. Hence, the comparison makes sense only for those
    // PackedCandidates that have ndof != 0.
    double diffNormalizedChi2 = 0;
    if(trackPc.ndof() != 0) {
      h_selectionFlow->Fill(sf_PCNdofNot0);
      diffNormalizedChi2 = trackPc.normalizedChi2() - track.normalizedChi2();
      fillNoFlow(h_diffNormalizedChi2, diffNormalizedChi2);
    }
    fillNoFlow(h_diffNdof, trackPc.ndof() - track.ndof());

    auto diffCharge = trackPc.charge() - track.charge();
    fillNoFlow(h_diffCharge, diffCharge);
    int diffHP = static_cast<int>(trackPc.quality(reco::TrackBase::highPurity)) - static_cast<int>(track.quality(reco::TrackBase::highPurity));
    fillNoFlow(h_diffIsHighPurity,  diffHP);

    /*
    edm::LogPrint("Foo") << "Track pt " << track.pt() << " PC " << trackPc.pt() << " diff " << (trackPc.pt()-track.pt())
                         << " ulps " << boost::math::float_distance(trackPc.pt(), track.pt())
                         << " relative " << (trackPc.pt()-track.pt())/track.pt()
                         << " mydiff " << ulpDiffRelative(trackPc.pt(), track.pt());
    */

    const auto diffPt = diffRelative(trackPc.pt(), track.pt());
    fillNoFlow(h_diffQoverp, diffRelative(trackPc.qoverp(), track.qoverp()));
    fillNoFlow(h_diffPt    , diffPt);
    fillNoFlow(h_diffEta   , diffRelative(trackPc.eta()   , track.eta()   ));
    fillNoFlow(h_diffTheta , diffRelative(trackPc.theta() , track.theta() ));
    fillNoFlow(h_diffPhi   , diffRelative(trackPc.phi()   , track.phi()   ));

    const auto diffDzPV = diffRelative(pcRef->dz(pv.position()), track.dz(pv.position()));
    const auto diffDzAssocPV = diffRelative(pcRef->dzAssociatedPV(), track.dz(pcVertex.position()));
    const auto diffDxyPV = diffRelative(pcRef->dxy(pv.position())    , track.dxy(pv.position()));
    const auto diffDxyAssocPV = diffRelative(pcRef->dxy()    , track.dxy(pcVertex.position()));
    fillNoFlow(h_diffDxyAssocPV, diffDxyAssocPV);
    fillNoFlow(h_diffDxyPV     , diffDxyPV);
    fillNoFlow(h_diffDzAssocPV , diffDzAssocPV);
    fillNoFlow(h_diffDzPV      , diffDzPV);
    fillNoFlow(h_diffTrackDxy  , diffRelative(trackPc.dxy()   , track.dxy()   ));
    fillNoFlow(h_diffTrackDz   , diffRelative(trackPc.dz()    , track.dz()    ));

    //const bool dzErrorFinite = isDzErrorPackedFinite(track.dzError()); // FIXME
    //const double diffDzError = dzErrorFinite ? diffRelative(pcRef->dzError(), track.dzError()) : 0; // FIXME 
    //const double diffQoverpError = diffRelative(trackPc.qoverpError(), track.qoverpError());
    const auto diffQoverpError = h_diffQoverpError.fill(trackPc.qoverpError(), track.qoverpError());

    const double diffThetaError = diffRelative(trackPc.thetaError() , track.thetaError());
    const double diffPhiError = diffRelative(trackPc.phiError()   , track.phiError());
    const bool dzErrorFinite = isDzErrorPackedFinite(track);
    const double diffDzError = dzErrorFinite ? diffRelative(pcRef->dzError(), track.dzError()) : 0;
    const double diffDszError = dzErrorFinite ? diffRelative(pcRef->dzError(), track.dszError()) : 0;
    const auto diffDxyError = diffRelative(pcRef->dxyError()  , track.dxyError());
    //fillNoFlow(h_diffQoverpError, diffQoverpError);
    fillNoFlow(h_diffPtError    , diffRelative(trackPc.ptError()    , track.ptError()    ));
    fillNoFlow(h_diffEtaError   , diffRelative(trackPc.etaError()   , track.etaError()   ));
    fillNoFlow(h_diffThetaError , diffThetaError);
    fillNoFlow(h_diffPhiError   , diffPhiError);
    fillNoFlow(h_diffDxyError   , diffDxyError);
    if(dzErrorFinite) fillNoFlow(h_diffDzError, diffDzError);
    if(dzErrorFinite) fillNoFlow(h_diffDszError, diffDszError);
    h_diffDzErrorInf->Fill(static_cast<int>(edm::isNotFinite(pcRef->dzError())) - static_cast<int>(!dzErrorFinite));
    fillNoFlow(h_diffTrackDxyError, diffRelative(trackPc.dxyError()  , track.dxyError()));
    if(dzErrorFinite) fillNoFlow(h_diffTrackDzError , diffRelative(trackPc.dzError(), track.dzError()));

    auto fillCov = [&](MonitorElement *me, const int i, const int j) {
      const auto diffRel = diffRelative(trackPc.covariance(i, j), track.covariance(i, j));
      fillNoFlow(me, diffRel);
      return diffRel;
    };
    /*
    fillCov(h_diffCovQoverpLambda, reco::TrackBase::i_qoverp, reco::TrackBase::i_lambda);
    fillCov(h_diffCovQoverpPhi,    reco::TrackBase::i_qoverp, reco::TrackBase::i_phi);
    fillCov(h_diffCovQoverpDxy,    reco::TrackBase::i_qoverp, reco::TrackBase::i_dxy);
    fillCov(h_diffCovQoverpDz,     reco::TrackBase::i_qoverp, reco::TrackBase::i_dsz);
    fillCov(h_diffCovLambdaPhi,    reco::TrackBase::i_lambda, reco::TrackBase::i_phi);
    fillCov(h_diffCovLambdaDxy,    reco::TrackBase::i_lambda, reco::TrackBase::i_dxy);
    */
    const auto diffCovLambdaDz = fillCov(h_diffCovLambdaDz,     reco::TrackBase::i_lambda, reco::TrackBase::i_dsz);
    const auto diffCovPhiDxy = fillCov(h_diffCovPhiDxy,       reco::TrackBase::i_phi,    reco::TrackBase::i_dxy);
    //fillCov(h_diffCovPhiDz,        reco::TrackBase::i_phi,    reco::TrackBase::i_dsz);
    const auto diffCovDxyDz = fillCov(h_diffCovDxyDz,        reco::TrackBase::i_dxy,    reco::TrackBase::i_dsz);

    // For the non-HitPattern ones, take into account the PackedCandidate packing precision
    const auto trackNumberOfHits = track.hitPattern().numberOfValidHits();
    const auto trackNumberOfPixelHits = track.hitPattern().numberOfValidPixelHits();
    const auto trackNumberOfStripHits = track.hitPattern().numberOfValidStripHits();
    const auto pcNumberOfHits = pcRef->numberOfHits();
    const auto pcNumberOfPixelHits = pcRef->numberOfPixelHits();
    const auto pcNumberOfStripHits = pcNumberOfHits - pcNumberOfPixelHits;

    const int pixelOverflow = trackNumberOfPixelHits > pat::PackedCandidate::trackPixelHitsMask ? trackNumberOfPixelHits - pat::PackedCandidate::trackPixelHitsMask : 0;
    const int stripOverflow = trackNumberOfStripHits > pat::PackedCandidate::trackStripHitsMask ? trackNumberOfStripHits - pat::PackedCandidate::trackStripHitsMask : 0;
    const int hitsOverflow = trackNumberOfHits > (pat::PackedCandidate::trackPixelHitsMask+pat::PackedCandidate::trackStripHitsMask) ? trackNumberOfHits - (pat::PackedCandidate::trackPixelHitsMask+pat::PackedCandidate::trackStripHitsMask) : 0;
    // PackedCandidate counts overflow pixel hits as strip
    const int pixelInducedStripOverflow = (trackNumberOfStripHits+pixelOverflow) > pat::PackedCandidate::trackStripHitsMask ? (trackNumberOfStripHits+pixelOverflow-stripOverflow) - pat::PackedCandidate::trackStripHitsMask : 0;
    h_numberPixelHitsOverMax->Fill(pixelOverflow);
    h_numberStripHitsOverMax->Fill(stripOverflow);
    h_numberHitsOverMax->Fill(hitsOverflow);

    int diffNumberOfPixelHits = 0;
    int diffNumberOfHits = 0;
    if(pixelOverflow) {
      diffNumberOfPixelHits = pcNumberOfPixelHits - pat::PackedCandidate::trackPixelHitsMask;
    }
    else {
      diffNumberOfPixelHits = pcNumberOfPixelHits - trackNumberOfPixelHits;
    }
    if(stripOverflow || pixelInducedStripOverflow || pixelOverflow) {
      int diffNumberOfStripHits = 0;
      if(stripOverflow || pixelInducedStripOverflow) {
        diffNumberOfStripHits = pcNumberOfStripHits - pat::PackedCandidate::trackStripHitsMask;
      }
      else if(pixelOverflow) {
        diffNumberOfStripHits = (pcNumberOfStripHits - pixelOverflow) - trackNumberOfStripHits;
      }

      diffNumberOfHits = diffNumberOfPixelHits + diffNumberOfStripHits;
      /*
      edm::LogWarning("Foo") << "pcNumberOfHits " << pcNumberOfHits
                             << " pcNumberOfPixelHits " << pcNumberOfPixelHits
                             << " pcNumberOfStripHits " << pcNumberOfStripHits
                             << " trackNumberOfHits " << trackNumberOfHits
                             << " trackNumberOfPixelHits " << trackNumberOfPixelHits
                             << " trackNumberOfStripHits " << trackNumberOfStripHits
                             << " pixelOverflow " << pixelOverflow
                             << " stripOverflow " << stripOverflow
                             << " pixelInducedStripOverflow " << pixelInducedStripOverflow
                             << " diffNumberOfPixelHits " << diffNumberOfPixelHits
                             << " diffNumberOfStripHits " << diffNumberOfStripHits
                             << " diffNumberOfHits " << diffNumberOfHits;
      */
    }
    else {
      diffNumberOfHits = pcNumberOfHits - trackNumberOfHits;
    }

    fillNoFlow(h_diffNumberOfPixelHits, diffNumberOfPixelHits);
    fillNoFlow(h_diffNumberOfHits, diffNumberOfHits);

    int diffLostInnerHits = 0;
    const auto trackLostInnerHits = track.hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);
    switch(pcRef->lostInnerHits()) {
    case pat::PackedCandidate::validHitInFirstPixelBarrelLayer:
    case pat::PackedCandidate::noLostInnerHits:
      diffLostInnerHits = -trackLostInnerHits;
      break;
    case pat::PackedCandidate::oneLostInnerHit:
      diffLostInnerHits = 1-trackLostInnerHits;
      break;
    case pat::PackedCandidate::moreLostInnerHits:
      diffLostInnerHits = trackLostInnerHits>=2 ? 0 : 2-trackLostInnerHits;
      break;
    }
    fillNoFlow(h_diffLostInnerHits, diffLostInnerHits);

    // For HitPattern ones, calculate the full diff (i.e. some differences are expected)
    auto diffHitPatternNumberOfValidPixelHits = trackPc.hitPattern().numberOfValidPixelHits() - trackNumberOfPixelHits;
    fillNoFlow(h_diffHitPatternNumberOfValidPixelHits, diffHitPatternNumberOfValidPixelHits);
    auto diffHitPatternNumberOfValidHits = trackPc.hitPattern().numberOfValidHits() - trackNumberOfHits;
    fillNoFlow(h_diffHitPatternNumberOfValidHits, diffHitPatternNumberOfValidHits);
    fillNoFlow(h_diffHitPatternNumberOfLostInnerHits, trackPc.hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS) - track.hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS));

    // hasValidHitInFirstPixelBarrel is set only if numberOfLostHits(MISSING_INNER_HITS) == 0
    int diffHitPatternHasValidHitInFirstPixelBarrel = 0;
    if(track.hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS) == 0) {
      h_selectionFlow->Fill(sf_NoMissingInnerHits);
      diffHitPatternHasValidHitInFirstPixelBarrel = static_cast<int>(trackPc.hitPattern().hasValidHitInFirstPixelBarrel()) - static_cast<int>(track.hitPattern().hasValidHitInFirstPixelBarrel());
      fillNoFlow(h_diffHitPatternHasValidHitInFirstPixelBarrel, diffHitPatternHasValidHitInFirstPixelBarrel);
    }

    // Print warning if there are differences outside the expected range
    if(diffNormalizedChi2 < -1 || diffNormalizedChi2 > 0 || diffCharge != 0 || diffHP != 0
       || diffNumberOfPixelHits != 0 || diffNumberOfHits != 0 || diffLostInnerHits != 0
       || diffHitPatternHasValidHitInFirstPixelBarrel != 0
       //|| std::abs(diffPt) > 0.2
       || std::abs(diffDzPV) > 0.01 || std::abs(diffDzAssocPV) > 0.01
       || std::abs(diffDxyPV) > 0.01 || std::abs(diffDxyAssocPV) > 0.01
       || std::abs(diffDszError) > 0.01 || std::abs(diffDxyError) > 0.01
       || diffQoverpError.outsideExpectedRange(0.0, 0.13)
       //|| diffQoverpError < 0.0 || diffQoverpError > 0.13
       || diffThetaError < 0.0 || diffThetaError > 0.13
       || diffPhiError < 0.0 || diffPhiError > 0.13
       || std::abs(diffCovDxyDz) > 0.05
       //|| std::abs(diffCovLambdaDz) > 0.1
       ) {

      edm::LogWarning("PackedCandidateTrackValidator") << "Track " << i << " pt " << track.pt() << " eta " << track.eta() << " phi " << track.phi() << " chi2 " << track.chi2() << " ndof " << track.ndof()
                                                       << "\n"
                                                       << "  ptError " << track.ptError() << " etaError " << track.etaError() << " phi " << track.phiError()
                                                       << "\n"
                                                       << "  refpoint " << track.referencePoint() << " momentum " << track.momentum()
                                                       << "\n"
                                                       << "  dxy " << track.dxy() << " dz " << track.dz()
                                                       << "\n"
                                                       << "  " << TrackAlgoPrinter(track)
                                                       << " lost inner hits " << trackLostInnerHits
                                                       << " lost outer hits " << track.hitPattern().numberOfLostHits(reco::HitPattern::MISSING_OUTER_HITS)
                                                       << " hitpattern " << HitPatternPrinter(track)
                                                       << " \n"
                                                       << " PC " << pcRef.id() << ":" << pcRef.key() << " track pt " << trackPc.pt() << " eta " << trackPc.eta() << " phi " << trackPc.phi() << " chi2 " << trackPc.chi2() << " ndof " << trackPc.ndof() << " pdgId " << pcRef->pdgId() << " mass " << pcRef->mass()
                                                       << "\n"
                                                       << "  ptError " << trackPc.ptError() << " etaError " << trackPc.etaError() << " phi " << trackPc.phiError()
                                                       << "\n"
                                                       << "  pc.vertex " << pcRef->vertex() << " momentum " << pcRef->momentum() << " track " << trackPc.momentum()
                                                       << "\n"
                                                       << "  dxy " << trackPc.dxy() << " dz " << trackPc.dz() << " pc.dz " << pcRef->dz()
                                                       << " dxyError " << trackPc.dxyError() << " dzError " << trackPc.dzError()
                                                       << "\n"
                                                       << " (diff PackedCandidate track)"
                                                       << " highPurity " << diffHP << " " << trackPc.quality(reco::TrackBase::highPurity) << " " << track.quality(reco::TrackBase::highPurity)
                                                       << " charge " << diffCharge << " " << trackPc.charge() << " " << track.charge()
                                                       << " normalizedChi2 " << diffNormalizedChi2 << " " << trackPc.normalizedChi2() << " " << track.normalizedChi2()
                                                       << "\n "
                                                       << " numberOfHits " << diffNumberOfHits << " " << pcNumberOfHits << " " << trackNumberOfHits
                                                       << " numberOfPixelHits " << diffNumberOfPixelHits << " " << pcNumberOfPixelHits << " " << trackNumberOfPixelHits
                                                       << " numberOfStripHits # " << pcNumberOfStripHits << " " << trackNumberOfStripHits
                                                       << "\n "
                                                       << " hitPattern.numberOfValidPixelHits " << diffHitPatternNumberOfValidPixelHits << " " << trackPc.hitPattern().numberOfValidPixelHits() << " " << track.hitPattern().numberOfValidPixelHits()
                                                       << " hitPattern.numberOfValidHits " << diffHitPatternNumberOfValidHits << " " << trackPc.hitPattern().numberOfValidHits() << " " << track.hitPattern().numberOfValidHits()
                                                       << " hitPattern.hasValidHitInFirstPixelBarrel " << diffHitPatternHasValidHitInFirstPixelBarrel << " " << trackPc.hitPattern().hasValidHitInFirstPixelBarrel() << " " << track.hitPattern().hasValidHitInFirstPixelBarrel()
                                                       << "\n "
                                                       << " lostInnerHits  " << diffLostInnerHits << " " << pcRef->lostInnerHits() << " #"
                                                       << "\n "
                                                       << " dz(PV) " << diffDzPV << " " << pcRef->dz(pv.position()) << " " << track.dz(pv.position())
                                                       << " dz(assocPV) " << diffDzAssocPV << " " << pcRef->dzAssociatedPV() << " " << track.dz(pcVertex.position())
                                                       << " dxy(PV) " << diffDxyPV << " " << pcRef->dxy(pv.position()) << " " << track.dxy(pv.position())
                                                       << " dxy(assocPV) " << diffDxyAssocPV << " " << pcRef->dxy() << " " << track.dxy(pcVertex.position())
                                                       << "\n "
        /*
                                                       << " dzError " << diffDzError << " " << trackPc.dzError() << " " << track.dzError() << " (" << packUnpack(track.dzError()) << ")"
                                                       << " dxyError " << diffDxyError << " " << trackPc.dxyError() << " " << track.dxyError();
        */
        //                                                << " qoverpError " << diffQoverpError << " " << trackPc.qoverpError() << " " << track.qoverpError()
                                                       << " qoverpError " << diffQoverpError
                                                       << " phiError " << diffPhiError << " " << trackPc.phiError() << " " << track.phiError()
                                                       << "\n "
                                                       << " thetaError " << diffThetaError << " " << trackPc.thetaError() << " " << track.thetaError() << "(" << track.covariance(reco::TrackBase::i_lambda, reco::TrackBase::i_lambda) << ")"
                                                       << "\n "
                                                       << " dszError " << diffDszError << " " << pcRef->dzError() << " " << track.dszError() << " (dz " << track.dzError() << ")"
                                                       << " dxyError " << diffDxyError << " " << pcRef->dxyError() << " " << track.dxyError()
                                                       << "\n "
                                                       << " cov(dxy, dz) " << diffCovDxyDz << " " << CovPrinter(trackPc, track, reco::TrackBase::i_dxy, reco::TrackBase::i_dsz, 10000.0)
                                                       << " cov(lambda, dz) " << diffCovLambdaDz << " " << CovPrinter(trackPc, track, reco::TrackBase::i_lambda, reco::TrackBase::i_dsz)
                                                       << " cov(phi, dxy) " << diffCovPhiDxy << " " << CovPrinter(trackPc, track, reco::TrackBase::i_phi, reco::TrackBase::i_dxy);

      /*
                                                       << "\n"
                                                       << " dz(PV) "
                                                       << "\n "
                                                       << " track " << DzCalculationPrinter(track.referencePoint(), pv.position(), track.momentum(), track.phi())
                                                       << "\n "
                                                       << " PC    " << DzCalculationPrinter(pcRef->vertex(), pv.position(), pcRef->momentum(), pcRef->phi())
                                                       << "\n "
                                                       << " PCvtx " << DzCalculationPrinter(pcRef->vertex(), pv.position(), pcRef->momentum(), pcRef->phiAtVtx())
                                                       << "\n "
                                                       << " dxy(PV) "
                                                       << "\n "
                                                       << " track " << DxyCalculationPrinter(track.referencePoint(), pv.position(), track.momentum(), track.phi())
                                                       << "\n "
                                                       << " PC    " << DxyCalculationPrinter(pcRef->vertex(), pv.position(), pcRef->momentum(), pcRef->phi())
                                                       << "\n "
                                                       << " PCvtx " << DxyCalculationPrinter(pcRef->vertex(), pv.position(), pcRef->momentum(), pcRef->phiAtVtx());
      */

      

      edm::LogWarning("PackedCandidateTrackValidator") << "Reco Primary vertex " << pv.position() << " associated PV " << pcVertex.position()
                                                       << "\n"
                                                       << "Packed Primary vertex " << slimmedPV.position() << " associated PV " << slimmedVertexRef->position();
        
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PackedCandidateTrackValidator);
