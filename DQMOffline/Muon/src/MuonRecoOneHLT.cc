#include "DQMOffline/Muon/src/MuonRecoOneHLT.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h" 
#include "DataFormats/MuonReco/interface/MuonEnergy.h"

#include "FWCore/Common/interface/TriggerNames.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"



#include <string>
#include "TMath.h"
using namespace std;
using namespace edm;

// Uncomment to DEBUG
//#define DEBUG

MuonRecoOneHLT::MuonRecoOneHLT(const edm::ParameterSet& pSet, MuonServiceProxy *theService):MuonAnalyzerBase(theService) {
  parameters = pSet;
  
  ParameterSet muonparms   = parameters.getParameter<edm::ParameterSet>("SingleMuonTrigger");
  ParameterSet dimuonparms = parameters.getParameter<edm::ParameterSet>("DoubleMuonTrigger");
  _SingleMuonEventFlag     = new GenericTriggerEventFlag( muonparms );
  _DoubleMuonEventFlag     = new GenericTriggerEventFlag( dimuonparms );
  
  // Trigger Expresions in case de connection to the DB fails
  singlemuonExpr_          = muonparms.getParameter<std::vector<std::string> >("hltPaths");
  doublemuonExpr_          = dimuonparms.getParameter<std::vector<std::string> >("hltPaths");
}


MuonRecoOneHLT::~MuonRecoOneHLT() {
  delete _SingleMuonEventFlag;
  delete _DoubleMuonEventFlag;
}
void MuonRecoOneHLT::beginJob(DQMStore * dbe) {
#ifdef DEBUG
  cout << "[MuonRecoOneHLT]  beginJob " << endl;
#endif
  dbe->setCurrentFolder("Muons/MuonRecoOneHLT");
  
  theMuonCollectionLabel = parameters.getParameter<edm::InputTag>("MuonCollection");
  vertexTag  = parameters.getParameter<edm::InputTag>("vertexLabel");
  bsTag  = parameters.getParameter<edm::InputTag>("bsLabel");

  
  muReco = dbe->book1D("Muon_Reco", "Muon Reconstructed Tracks", 6, 1, 7);
  muReco->setBinLabel(1,"glb+tk+sta"); 
  muReco->setBinLabel(2,"glb+sta");
  muReco->setBinLabel(3,"tk+sta");
  muReco->setBinLabel(4,"tk");
  muReco->setBinLabel(5,"sta");
  muReco->setBinLabel(6,"calo");

  // monitoring of eta parameter
  etaBin = parameters.getParameter<int>("etaBin");
  etaMin = parameters.getParameter<double>("etaMin");
  etaMax = parameters.getParameter<double>("etaMax");
  
  std::string histname = "GlbMuon_";
  etaGlbTrack.push_back(dbe->book1D(histname+"Glb_eta", "#eta_{GLB}", etaBin, etaMin, etaMax));
  etaGlbTrack.push_back(dbe->book1D(histname+"Tk_eta", "#eta_{TKfromGLB}", etaBin, etaMin, etaMax));
  etaGlbTrack.push_back(dbe->book1D(histname+"Sta_eta", "#eta_{STAfromGLB}", etaBin, etaMin, etaMax));
  etaTight = dbe->book1D("TightMuon_eta", "#eta_{GLB}", etaBin, etaMin, etaMax);
  etaTrack = dbe->book1D("TkMuon_eta", "#eta_{TK}", etaBin, etaMin, etaMax);
  etaStaTrack = dbe->book1D("StaMuon_eta", "#eta_{STA}", etaBin, etaMin, etaMax);

  // monitoring of phi paramater
  phiBin = parameters.getParameter<int>("phiBin");
  phiMin = parameters.getParameter<double>("phiMin");
  phiMax = parameters.getParameter<double>("phiMax");
  phiGlbTrack.push_back(dbe->book1D(histname+"Glb_phi", "#phi_{GLB}", phiBin, phiMin, phiMax));
  phiGlbTrack[0]->setAxisTitle("rad");
  phiGlbTrack.push_back(dbe->book1D(histname+"Tk_phi", "#phi_{TKfromGLB}", phiBin, phiMin, phiMax));
  phiGlbTrack[1]->setAxisTitle("rad");
  phiGlbTrack.push_back(dbe->book1D(histname+"Sta_phi", "#phi_{STAfromGLB}", phiBin, phiMin, phiMax));
  phiGlbTrack[2]->setAxisTitle("rad");
  phiTight = dbe->book1D("TightMuon_phi", "#phi_{GLB}", phiBin, phiMin, phiMax);
  phiTrack = dbe->book1D("TkMuon_phi", "#phi_{TK}", phiBin, phiMin, phiMax);
  phiTrack->setAxisTitle("rad"); 
  phiStaTrack = dbe->book1D("StaMuon_phi", "#phi_{STA}", phiBin, phiMin, phiMax);
  phiStaTrack->setAxisTitle("rad"); 
  
  // monitoring of the chi2 parameter
  chi2Bin = parameters.getParameter<int>("chi2Bin");
  chi2Min = parameters.getParameter<double>("chi2Min");
  chi2Max = parameters.getParameter<double>("chi2Max");
  chi2OvDFGlbTrack.push_back(dbe->book1D(histname+"Glb_chi2OverDf", "#chi_{2}OverDF_{GLB}", chi2Bin, chi2Min, chi2Max));
  chi2OvDFGlbTrack.push_back(dbe->book1D(histname+"Tk_chi2OverDf",  "#chi_{2}OverDF_{TKfromGLB}", phiBin, chi2Min, chi2Max));
  chi2OvDFGlbTrack.push_back(dbe->book1D(histname+"Sta_chi2OverDf", "#chi_{2}OverDF_{STAfromGLB}", chi2Bin, chi2Min, chi2Max));
  chi2OvDFTight    = dbe->book1D("TightMuon_chi2OverDf", "#chi_{2}OverDF_{GLB}", chi2Bin, chi2Min, chi2Max);
  chi2OvDFTrack    = dbe->book1D("TkMuon_chi2OverDf",    "#chi_{2}OverDF_{TK}", chi2Bin, chi2Min, chi2Max);
  chi2OvDFStaTrack = dbe->book1D("StaMuon_chi2OverDf",   "#chi_{2}OverDF_{STA}", chi2Bin, chi2Min, chi2Max);

  // monitoring of the transverse momentum
  ptBin = parameters.getParameter<int>("ptBin");
  ptMin = parameters.getParameter<double>("ptMin");
  ptMax = parameters.getParameter<double>("ptMax");
  ptGlbTrack.push_back(dbe->book1D(histname+"Glb_pt", "pt_{GLB}", ptBin, ptMin, ptMax));
  ptGlbTrack[0]->setAxisTitle("GeV"); 
  ptGlbTrack.push_back(dbe->book1D(histname+"Tk_pt", "pt_{TKfromGLB}", ptBin, ptMin, ptMax));
  ptGlbTrack[1]->setAxisTitle("GeV"); 
  ptGlbTrack.push_back(dbe->book1D(histname+"Sta_pt", "pt_{STAfromGLB}", ptBin, ptMin, ptMax));
  ptGlbTrack[2]->setAxisTitle("GeV"); 
  ptTight = dbe->book1D("TightMuon_pt", "pt_{GLB}", ptBin, ptMin, ptMax);
  ptTight->setAxisTitle("GeV"); 
  ptTrack = dbe->book1D("TkMuon_pt", "pt_{TK}", ptBin, ptMin, ptMax);
  ptTrack->setAxisTitle("GeV"); 
  ptStaTrack = dbe->book1D("StaMuon_pt", "pt_{STA}", ptBin, ptMin, ptMax);
  ptStaTrack->setAxisTitle("GeV"); 
}

void MuonRecoOneHLT::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup){
#ifdef DEBUG
  cout << "[MuonRecoOneHLT]  beginRun " << endl;
  cout << "[MuonRecoOneHLT]  Is MuonEventFlag On? "<< _SignleMuonEventFlag->on() << endl;
#endif
  if ( _SingleMuonEventFlag->on() ) _SingleMuonEventFlag->initRun( iRun, iSetup );
  if ( _DoubleMuonEventFlag->on() ) _DoubleMuonEventFlag->initRun( iRun, iSetup );

  if (_SingleMuonEventFlag->on() && _SingleMuonEventFlag->expressionsFromDB(_SingleMuonEventFlag->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
    singlemuonExpr_ = _SingleMuonEventFlag->expressionsFromDB(_SingleMuonEventFlag->hltDBKey(),iSetup);
  if (_DoubleMuonEventFlag->on() && _DoubleMuonEventFlag->expressionsFromDB(_DoubleMuonEventFlag->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
    singlemuonExpr_ = _DoubleMuonEventFlag->expressionsFromDB(_DoubleMuonEventFlag->hltDBKey(),iSetup);
}
void MuonRecoOneHLT::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, 
			     //			     const reco::Muon& recoMu, 
			     const edm::TriggerResults& triggerResults) {
#ifdef DEBUG
  cout << "[MuonRecoOneHLT]  analyze "<< endl;
#endif


  // ==========================================================
  // Look for the Primary Vertex (and use the BeamSpot instead, if you can't find it):

  reco::Vertex::Point posVtx;
  reco::Vertex::Error errVtx;
 
  unsigned int theIndexOfThePrimaryVertex = 999.;
 
  edm::Handle<reco::VertexCollection> vertex;
  iEvent.getByLabel(vertexTag, vertex);

  if ( vertex.isValid() ){
  for (unsigned int ind=0; ind<vertex->size(); ++ind) {
    if ( (*vertex)[ind].isValid() && !((*vertex)[ind].isFake()) ) {
      theIndexOfThePrimaryVertex = ind;
      break;
    }
  }
  }
  if (theIndexOfThePrimaryVertex<100) {
    posVtx = ((*vertex)[theIndexOfThePrimaryVertex]).position();
    errVtx = ((*vertex)[theIndexOfThePrimaryVertex]).error();
  }   else {
    LogInfo("RecoMuonValidator") << "reco::PrimaryVertex not found, use BeamSpot position instead\n";
  
    edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
    iEvent.getByLabel(bsTag,recoBeamSpotHandle);
    
    reco::BeamSpot bs = *recoBeamSpotHandle;
    
    posVtx = bs.position();
    errVtx(0,0) = bs.BeamWidthX();
    errVtx(1,1) = bs.BeamWidthY();
    errVtx(2,2) = bs.sigmaZ();
  }
  const reco::Vertex thePrimaryVertex(posVtx,errVtx);

  // ==========================================================



  
  //  TEST FOR ONLY TAKE HIGHEST PT MUON
  edm::Handle<reco::MuonCollection> muons;
  iEvent.getByLabel(theMuonCollectionLabel,muons);


  std::map<float,reco::Muon> muonMap;
  for (reco::MuonCollection::const_iterator recoMu = muons->begin(); recoMu!=muons->end(); ++recoMu){
    muonMap[recoMu->pt()] = *recoMu;
  }
  std::vector<reco::Muon> LeadingMuon;
  for( std::map<float,reco::Muon>::reverse_iterator rit=muonMap.rbegin(); rit!=muonMap.rend(); ++rit){
    LeadingMuon.push_back( (*rit).second );
  }

  reco::BeamSpot beamSpot;
  Handle<reco::BeamSpot> beamSpotHandle;
  iEvent.getByLabel("offlineBeamSpot", beamSpotHandle);
  beamSpot = *beamSpotHandle;
  
  const edm::TriggerNames& triggerNames = iEvent.triggerNames(triggerResults);
  const unsigned int nTrig(triggerNames.size());
  bool _trig_SingleMu = false;
  bool _trig_DoubleMu = false;
  for (unsigned int i=0;i<nTrig;++i){
    if (triggerNames.triggerName(i).find(singlemuonExpr_[0].substr(0,singlemuonExpr_[0].rfind("_v")+2))!=std::string::npos && triggerResults.accept(i))
      _trig_SingleMu = true;
    if (triggerNames.triggerName(i).find(doublemuonExpr_[0].substr(0,doublemuonExpr_[0].rfind("_v")+2))!=std::string::npos && triggerResults.accept(i))
      _trig_DoubleMu = true;
  }
#ifdef DEBUG
  cout << "[MuonRecoOneHLT]  Trigger Fired ? "<< _trig_SingleMu << endl;
#endif

  if (!_trig_SingleMu && !_trig_DoubleMu) return;
  if (LeadingMuon.size() == 0)            return; 
  //  if (_MuonEventFlag->on() && !(_MuonEventFlag->accept(iEvent,iSetup))) return;

  // Check if Muon is Global
  if(LeadingMuon[0].isGlobalMuon()) {
    LogTrace(metname)<<"[MuonRecoOneHLT] The mu is global - filling the histos";
    if(LeadingMuon[0].isTrackerMuon() && LeadingMuon[0].isStandAloneMuon())          muReco->Fill(1);
    if(!(LeadingMuon[0].isTrackerMuon()) && LeadingMuon[0].isStandAloneMuon())       muReco->Fill(2);
    if(!LeadingMuon[0].isStandAloneMuon())   
      LogTrace(metname)<<"[MuonRecoOneHLT] ERROR: the mu is global but not standalone!";

    // get the track combinig the information from both the Tracker and the Spectrometer
    reco::TrackRef recoCombinedGlbTrack = LeadingMuon[0].combinedMuon();
    // get the track using only the tracker data
    reco::TrackRef recoTkGlbTrack = LeadingMuon[0].track();
    // get the track using only the mu spectrometer data
    reco::TrackRef recoStaGlbTrack = LeadingMuon[0].standAloneMuon();

    etaGlbTrack[0]->Fill(recoCombinedGlbTrack->eta());
    etaGlbTrack[1]->Fill(recoTkGlbTrack->eta());
    etaGlbTrack[2]->Fill(recoStaGlbTrack->eta());
    
    phiGlbTrack[0]->Fill(recoCombinedGlbTrack->phi());
    phiGlbTrack[1]->Fill(recoTkGlbTrack->phi());
    phiGlbTrack[2]->Fill(recoStaGlbTrack->phi());
    
    chi2OvDFGlbTrack[0]->Fill(recoCombinedGlbTrack->normalizedChi2());
    chi2OvDFGlbTrack[1]->Fill(recoTkGlbTrack->normalizedChi2());
    chi2OvDFGlbTrack[2]->Fill(recoStaGlbTrack->normalizedChi2());

    ptGlbTrack[0]->Fill(recoCombinedGlbTrack->pt());
    ptGlbTrack[1]->Fill(recoTkGlbTrack->pt());
    ptGlbTrack[2]->Fill(recoStaGlbTrack->pt());
  }
  // Check if Muon is Tight
  if (muon::isTightMuon(LeadingMuon[0], thePrimaryVertex) ) { 
    
    LogTrace(metname)<<"[MuonRecoOneHLT] The mu is tracker only - filling the histos";
    
    reco::TrackRef recoCombinedGlbTrack = LeadingMuon[0].combinedMuon();

    etaTight->Fill(recoCombinedGlbTrack->eta());
    phiTight->Fill(recoCombinedGlbTrack->phi());
    chi2OvDFTight->Fill(recoCombinedGlbTrack->normalizedChi2());
    ptTight->Fill(recoCombinedGlbTrack->pt());
  }
  
  // Check if Muon is Tracker but NOT Global
  if(LeadingMuon[0].isTrackerMuon() && !(LeadingMuon[0].isGlobalMuon())) {
    LogTrace(metname)<<"[MuonRecoOneHLT] The mu is tracker only - filling the histos";
    if(LeadingMuon[0].isStandAloneMuon())          muReco->Fill(3);
    if(!(LeadingMuon[0].isStandAloneMuon()))        muReco->Fill(4);
    
    // get the track using only the tracker data
    reco::TrackRef recoTrack = LeadingMuon[0].track();

    etaTrack->Fill(recoTrack->eta());
    phiTrack->Fill(recoTrack->phi());
    chi2OvDFTrack->Fill(recoTrack->normalizedChi2());
    ptTrack->Fill(recoTrack->pt());
  }
    
  // Check if Muon is STA but NOT Global
  if(LeadingMuon[0].isStandAloneMuon() && !(LeadingMuon[0].isGlobalMuon())) {
    LogTrace(metname)<<"[MuonRecoOneHLT] The mu is STA only - filling the histos";
    if(!(LeadingMuon[0].isTrackerMuon()))         muReco->Fill(5);
     
    // get the track using only the mu spectrometer data
    reco::TrackRef recoStaTrack = LeadingMuon[0].standAloneMuon();

    etaStaTrack->Fill(recoStaTrack->eta());
    phiStaTrack->Fill(recoStaTrack->phi());
    chi2OvDFStaTrack->Fill(recoStaTrack->normalizedChi2());
    ptStaTrack->Fill(recoStaTrack->pt());
  }
  // Check if Muon is Only CaloMuon
  if(LeadingMuon[0].isCaloMuon() && !(LeadingMuon[0].isGlobalMuon()) && !(LeadingMuon[0].isTrackerMuon()) && !(LeadingMuon[0].isStandAloneMuon()))
    muReco->Fill(6);
}
