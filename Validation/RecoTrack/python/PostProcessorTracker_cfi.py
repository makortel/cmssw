import FWCore.ParameterSet.Config as cms

postProcessorTrack = cms.EDAnalyzer("DQMGenericClient",
    subDirs = cms.untracked.vstring("Tracking/Track/*"),
    efficiency = cms.vstring(
    "effic 'Efficiency vs #eta' num_assoc(simToReco)_eta num_simul_eta",
    "efficPt 'Efficiency vs p_{T}' num_assoc(simToReco)_pT num_simul_pT",
    "effic_vs_hit 'Efficiency vs hit' num_assoc(simToReco)_hit num_simul_hit",
    "effic_vs_phi 'Efficiency vs #phi' num_assoc(simToReco)_phi num_simul_phi",
    "effic_vs_dxy 'Efficiency vs Dxy' num_assoc(simToReco)_dxy num_simul_dxy",
    "effic_vs_dz 'Efficiency vs Dz' num_assoc(simToReco)_dz num_simul_dz",
    "duplicatesRate 'Duplicates Rate vs #eta' num_duplicate_eta num_reco_eta",
    "duplicatesRate_Pt 'Duplicates Rate vs p_{T}' num_duplicate_pT num_reco_pT",
    "duplicatesRate_hit 'Duplicates Rate vs hit' num_duplicate_hit num_reco_hit",
    "duplicatesRate_phi 'Duplicates Rate vs #phi' num_duplicate_phi num_reco_phi",
    "duplicatesRate_dxy 'Duplicates Rate vs Dxy' num_duplicate_dxy num_reco_dxy",
    "duplicatesRate_dz 'Duplicates Rate vs Dz' num_duplicate_dz num_reco_dz",
    "chargeMisIdRate 'Charge MisID Rate vs #eta' num_chargemisid_eta num_reco_eta",
    "chargeMisIdRate_Pt 'Charge MisID Rate vs p_{T}' num_chargemisid_pT num_reco_pT",
    "chargeMisIdRate_hit 'Charge MisID Rate vs hit' num_chargemisid_hit num_reco_hit",
    "chargeMisIdRate_phi 'Charge MisID Rate vs #phi' num_chargemisid_phi num_reco_phi",
    "chargeMisIdRate_dxy 'Charge MisID Rate vs Dxy' num_chargemisid_dxy num_reco_dxy",
    "chargeMisIdRate_dz 'Charge MisID Rate vs Dz' num_chargemisid_versus_dz num_reco_dz",
    "effic_vs_vertpos 'Efficiency vs vertpos' num_assoc(simToReco)_vertpos num_simul_vertpos",
    "effic_vs_zpos 'Efficiency vs zpos' num_assoc(simToReco)_zpos num_simul_zpos",
    "effic_vertcount_barrel 'efficiency in barrel vs N of pileup vertices' num_assoc(simToReco)_vertcount_barrel num_simul_vertcount_barrel",
    "effic_vertcount_fwdpos 'efficiency in endcap(+) vs N of pileup vertices' num_assoc(simToReco)_vertcount_fwdpos num_simul_vertcount_fwdpos",
    "effic_vertcount_fwdneg 'efficiency in endcap(-) vs N of pileup vertices' num_assoc(simToReco)_vertcount_fwdneg num_simul_vertcount_fwdneg",
    "effic_vertz_barrel 'efficiency in barrel vs z of primary interaction vertex' num_assoc(simToReco)_vertz_barrel num_simul_vertz_barrel",
    "effic_vertz_fwdpos 'efficiency in endcap(+) vs z of primary interaction vertex' num_assoc(simToReco)_vertz_fwdpos num_simul_vertz_fwdpos",
    "effic_vertz_fwdneg 'efficiency in endcap(-) vs z of primary interaction vertex' num_assoc(simToReco)_vertz_fwdneg num_simul_vertz_fwdneg",
    "pileuprate 'Pileup Rate vs #eta' num_pileup_eta num_reco_eta",
    "pileuprate_Pt 'Pileup rate vs p_{T}' num_pileup_pT num_reco_pT",
    "pileuprate_hit 'Pileup rate vs hit' num_pileup_hit num_reco_hit",
    "pileuprate_phi 'Pileup rate vs #phi' num_pileup_phi num_reco_phi",
    "pileuprate_dxy 'Pileup rate vs dxy' num_pileup_dxy num_reco_dxy",
    "pileuprate_dz 'Pileup rate vs dz' num_pileup_dz num_reco_dz",
    "fakerate 'Fake rate vs #eta' num_assoc(recoToSim)_eta num_reco_eta fake",
    "fakeratePt 'Fake rate vs p_{T}' num_assoc(recoToSim)_pT num_reco_pT fake",
    "fakerate_vs_hit 'Fake rate vs hit' num_assoc(recoToSim)_hit num_reco_hit fake",
    "fakerate_vs_phi 'Fake rate vs phi' num_assoc(recoToSim)_phi num_reco_phi fake",
    "fakerate_vs_dxy 'Fake rate vs dxy' num_assoc(recoToSim)_dxy num_reco_dxy fake",
    "fakerate_vs_dz 'Fake rate vs dz' num_assoc(recoToSim)_dz num_reco_dz fake",
    "fakerate_vertcount_barrel 'fake rate in barrel vs N of pileup vertices' num_assoc(recoToSim)_vertcount_barrel num_reco_vertcount_barrel fake",
    "fakerate_vertcount_fwdpos 'fake rate in endcap(+) vs N of pileup vertices' num_assoc(recoToSim)_vertcount_fwdpos num_reco_vertcount_fwdpos fake",
    "fakerate_vertcount_fwdneg 'fake rate in endcap(-) vs N of pileup vertices' num_assoc(recoToSim)_vertcount_fwdneg num_reco_vertcount_fwdneg fake"
    "fakerate_ootpu_entire 'fake rate from out of time pileup vs N of pileup vertices' num_assoc(recoToSim)_ootpu_entire num_reco_ootpu_entire",
    "fakerate_ootpu_barrel 'fake rate from out of time pileup in barrel vs N of pileup vertices' num_assoc(recoToSim)_ootpu_barrel num_reco_ootpu_barrel",
    "fakerate_ootpu_fwdpos 'fake rate from out of time pileup in endcap(+) vs N of pileup vertices' num_assoc(recoToSim)_ootpu_fwdpos num_reco_ootpu_fwdpos",
    "fakerate_ootpu_fwdneg 'fake rate from out of time pileup in endcap(-) vs N of pileup vertices' num_assoc(recoToSim)_ootpu_fwdneg num_reco_ootpu_fwdneg",

    "effic_vs_dzpvcut 'Efficiency vs. dz (PV)' num_assoc(simToReco)_dzpvcut num_simul_dzpvcut",
    "effic_vs_dzpvcut2 'Efficiency (tracking eff factorized out) vs. dz (PV)' num_assoc(simToReco)_dzpvcut num_simul2_dzpvcut",
    "fakerate_vs_dzpvcut 'Fake rate vs. dz(PV)' num_assoc(recoToSim)_dzpvcut num_reco_dzpvcut fake",
    "pileuprate_dzpvcut 'Pileup rate vs. dz(PV)' num_pileup_dzpvcut num_reco_dzpvcut",

    "effic_vs_dzpvcut_pt 'Fraction of true p_{T} carried by recoed TPs from PV vs. dz(PV)' num_assoc(simToReco)_dzpvcut_pt num_simul_dzpvcut_pt",
    "effic_vs_dzpvcut2_pt 'Fraction of true p_{T} carried by recoed TPs from PV (tracking eff factorized out) vs. dz(PV)' num_assoc(simToReco)_dzpvcut_pt num_simul2_dzpvcut_pt",
    "fakerate_vs_dzpvcut_pt 'Fraction of fake p_{T} carried by tracks from PV vs. dz(PV)' num_assoc(recoToSim)_dzpvcut_pt num_reco_dzpvcut_pt fake",
    "pileuprate_dzpvcut_pt 'Fraction of pileup p_{T} carried by tracks from PV vs. dz(PV)' num_pileup_dzpvcut_pt num_reco_dzpvcut_pt",

    "effic_vs_dzpvsigcut 'Efficiency vs. dz(PV)/dzError' num_assoc(simToReco)_dzpvsigcut num_simul_dzpvsigcut",
    "effic_vs_dzpvsigcut2 'Efficiency (tracking eff factorized out) vs. dz(PV)/dzError' num_assoc(simToReco)_dzpvsigcut num_simul2_dzpvsigcut",
    "fakerate_vs_dzpvsigcut 'Fake rate vs. dz(PV)/dzError' num_assoc(recoToSim)_dzpvsigcut num_reco_dzpvsigcut fake",
    "pileuprate_dzpvsigcut 'Pileup rate vs. dz(PV)/dzError' num_pileup_dzpvsigcut num_reco_dzpvsigcut",

    "effic_vs_dzpvsigcut_pt 'Fraction of true p_{T} carried by recoed TPs from PV vs. dz(PV)/dzError' num_assoc(simToReco)_dzpvsigcut_pt num_simul_dzpvsigcut_pt",
    "effic_vs_dzpvsigcut2_pt 'Fraction of true p_{T} carried by recoed TPs from PV (tracking eff factorized out) vs. dz(PV)/dzError' num_assoc(simToReco)_dzpvsigcut_pt num_simul2_dzpvsigcut_pt",
    "fakerate_vs_dzpvsigcut_pt 'Fraction of fake p_{T} carried by tracks from PV vs. dz(PV)/dzError' num_assoc(recoToSim)_dzpvsigcut_pt num_reco_dzpvsigcut_pt fake",
    "pileuprate_dzpvsigcut_pt 'Fraction of pileup p_{T} carried by tracks from PV vs. dz(PV)/dzError' num_pileup_dzpvsigcut_pt num_reco_dzpvsigcut_pt",
    ),
    resolution = cms.vstring(
                             "cotThetares_vs_eta '#sigma(cot(#theta)) vs #eta' cotThetares_vs_eta",
                             "cotThetares_vs_pt '#sigma(cot(#theta)) vs p_{T}' cotThetares_vs_pt",
                             "h_dxypulleta 'd_{xy} Pull vs #eta' dxypull_vs_eta",
                             "dxyres_vs_eta '#sigma(d_{xy}) vs #eta' dxyres_vs_eta",
                             "dxyres_vs_pt '#sigma(d_{xy}) vs p_{T}' dxyres_vs_pt",
                             "h_dzpulleta 'd_{z} Pull vs #eta' dzpull_vs_eta",
                             "dzres_vs_eta '#sigma(d_{z}) vs #eta' dzres_vs_eta",
                             "dzres_vs_pt '#sigma(d_{z}) vs p_{T}' dzres_vs_pt",
                             "etares_vs_eta '#sigma(#eta) vs #eta' etares_vs_eta",
                             "h_phipulleta '#phi Pull vs #eta' phipull_vs_eta",
                             "h_phipullphi '#phi Pull vs #phi' phipull_vs_phi",
                             "phires_vs_eta '#sigma(#phi) vs #eta' phires_vs_eta",
                             "phires_vs_phi '#sigma(#phi) vs #phi' phires_vs_phi",
                             "phires_vs_pt '#sigma(#phi) vs p_{T}' phires_vs_pt",
                             "h_ptpulleta 'p_{T} Pull vs #eta' ptpull_vs_eta",
                             "h_ptpullphi 'p_{T} Pull vs #phi' ptpull_vs_phi",
                             "ptres_vs_eta '#sigma(p_{T}) vs #eta' ptres_vs_eta",
                             "ptres_vs_phi '#sigma(p_{T}) vs #phi' ptres_vs_phi",
                             "ptres_vs_pt '#sigma(p_{T}) vs p_{T}' ptres_vs_pt",
                             "h_thetapulleta '#theta Pull vs #eta' thetapull_vs_eta",
                             "h_thetapullphi '#theta Pull vs #phi' thetapull_vs_phi"
                             ),
    profile= cms.vstring(
                         "chi2mean 'mean #chi^{2} vs #eta' chi2_vs_eta",
                         "chi2mean_vs_phi 'mean #chi^{2} vs #phi' chi2_vs_phi",
                         "chi2mean_vs_nhits 'mean #chi^{2} vs n. hits' chi2_vs_nhits",
                         "hits_eta 'mean #hits vs eta' nhits_vs_eta",
                         "hits_phi 'mean #hits vs #phi' nhits_vs_phi",
                         "losthits_eta 'mean #lost hits vs #eta' nlosthits_vs_eta",
			 "PXBhits_eta 'mean # PXB hits vs #eta' nPXBhits_vs_eta",
			 "PXFhits_eta 'mean # PXF hits vs #eta' nPXFhits_vs_eta",
			 "TIBhits_eta 'mean # TIB hits vs #eta' nTIBhits_vs_eta",
			 "TIDhits_eta 'mean # TID hits vs #eta' nTIDhits_vs_eta",
			 "TOBhits_eta 'mean # TOB hits vs #eta' nTOBhits_vs_eta",
			 "TEChits_eta 'mean # TEC hits vs #eta' nTEChits_vs_eta",
			 "LayersWithMeas_eta 'mean # LayersWithMeas vs #eta' nLayersWithMeas_vs_eta",
			 "PXLlayersWith2dMeas 'mean # PXLlayersWithMeas vs #eta' nPXLlayersWith2dMeas",
			 "STRIPlayersWithMeas_eta 'mean # STRIPlayersWithMeas vs #eta' nSTRIPlayersWithMeas_eta",
			 "STRIPlayersWith1dMeas_eta 'mean # STRIPlayersWith1dMeas vs #eta' nSTRIPlayersWith1dMeas_eta",
			 "STRIPlayersWith2dMeas_eta 'mean # STRIPlayersWith2dMeas vs #eta' nSTRIPlayersWith2dMeas_eta"
                         ),
    outputFileName = cms.untracked.string("")
)
