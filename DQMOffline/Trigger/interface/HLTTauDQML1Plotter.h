// -*- c++ -*-
#ifndef HLTTauDQML1Plotter_h
#define HLTTauDQML1Plotter_h

#include "DQMOffline/Trigger/interface/HLTTauDQMPlotter.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"

class HLTTauDQML1Plotter: private HLTTauDQMPlotter {
public:
    HLTTauDQML1Plotter(const edm::ParameterSet&, int, int, int, double, bool, double, const std::string& dqmBaseFolder);
    ~HLTTauDQML1Plotter();

    using HLTTauDQMPlotter::isValid;

    void beginRun();
    void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, const HLTTauDQMOfflineObjects& refC);

private:
    //The filters
    edm::InputTag l1ExtraTaus_;
    edm::InputTag l1ExtraJets_;

    bool doRefAnalysis_;
    double matchDeltaR_;
    double l1JetMinEt_;

    double maxEt_;
    int binsEt_;
    int binsEta_;
    int binsPhi_;

    //MonitorElements general
    MonitorElement* l1tauEt_;
    MonitorElement* l1tauEta_;
    MonitorElement* l1tauPhi_;

    MonitorElement* l1jetEt_;
    MonitorElement* l1jetEta_;
    MonitorElement* l1jetPhi_;

    //Monitor Elements for matching
    MonitorElement* l1tauEtRes_;
    MonitorElement* l1jetEtRes_;

    MonitorElement* l1tauEtEffNum_;
    MonitorElement* l1tauEtEffDenom_;

    MonitorElement* l1tauEtaEffNum_;
    MonitorElement* l1tauEtaEffDenom_;

    MonitorElement* l1tauPhiEffNum_;
    MonitorElement* l1tauPhiEffDenom_;

    MonitorElement* l1jetEtEffNum_;
    MonitorElement* l1jetEtEffDenom_;

    MonitorElement* l1jetEtaEffNum_;
    MonitorElement* l1jetEtaEffDenom_;

    MonitorElement* l1jetPhiEffNum_;
    MonitorElement* l1jetPhiEffDenom_;

    MonitorElement* firstTauEt_;
    MonitorElement* firstTauEta_;
    MonitorElement* firstTauPhi_;

    MonitorElement* secondTauEt_;
    MonitorElement* secondTauEta_;
    MonitorElement* secondTauPhi_;
};
#endif
