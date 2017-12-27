#include "CommonTools/CandAlgos/interface/GenJetParticleSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "SimGeneral/HepPDTRecord/interface/PdtEntry.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <algorithm>
using namespace std;
using namespace edm;

GenJetParticleSelector::GenJetParticleSelector(const ParameterSet& cfg, edm::ConsumesCollector & iC) :
  stableOnly_(cfg.getParameter<bool>("stableOnly")),
  partons_(false), bInclude_(false) {
  // TODO: the parameter retrieval can be simplified now that we have fillDescriptions
  const string excludeString("excludeList");
  const string includeString("includeList");
  vpdt includeList, excludeList;
  vector<string> vPdtParams = cfg.getParameterNamesForType<vpdt>();
  bool found = std::find(vPdtParams.begin(), vPdtParams.end(), includeString) != vPdtParams.end();
  if(found) includeList = cfg.getParameter<vpdt>(includeString);
  found = find(vPdtParams.begin(), vPdtParams.end(), excludeString) != vPdtParams.end();
  if(found) excludeList = cfg.getParameter<vpdt>(excludeString);
  const string partonsString("partons");
  vector<string> vBoolParams = cfg.getParameterNamesForType<bool>();
  found = find(vBoolParams.begin(), vBoolParams.end(), partonsString) != vBoolParams.end();
  if(found) partons_ = cfg.getParameter<bool>(partonsString);
  bool bExclude = false;
  if (!includeList.empty()) bInclude_ = true;
  if (!excludeList.empty()) bExclude = true;

  if (bInclude_ && bExclude) {
    throw cms::Exception("ConfigError", "not allowed to use both includeList and excludeList at the same time\n");
  }
  else if (bInclude_) {
    pdtList_ = includeList;
  }
  else {
    pdtList_ = excludeList;
  }
  if(stableOnly_ && partons_) {
    throw cms::Exception("ConfigError", "not allowed to have both stableOnly and partons true at the same time\n");
  }
}

void GenJetParticleSelector::fillPSetDescription(edm::ParameterSetDescription& desc) {
  desc.add<bool>("stableOnly", false);

  // PdtEntry can be constructed either from string or int32 (see
  // PdtEntry.h and .cc. I don't know if we can specialize
  // ParameterSetDescription similarly.
  desc.addNode(edm::ParameterDescription<std::vector<std::string> >("includeList", std::vector<std::string>{}, true) xor
               edm::ParameterDescription<std::vector<int> >("includeList", std::vector<int>{}, true) );
  desc.addNode(edm::ParameterDescription<std::vector<std::string> >("excludeList", std::vector<std::string>{}, true) xor
               edm::ParameterDescription<std::vector<int> >("excludeList", std::vector<int>{}, true) );

  desc.add<bool>("partons", false);
}

bool GenJetParticleSelector::operator()(const reco::Candidate& p) {
  int status = p.status();
  int id = abs(p.pdgId());
  if((!stableOnly_ || status == 1) && !partons_ &&
     ( (pIds_.find(id) == pIds_.end()) ^ bInclude_))
    return true;
  else if(partons_ &&
	  (p.numberOfDaughters() > 0 && (p.daughter(0)->pdgId() == 91 || p.daughter(0)->pdgId() == 92)) &&
	  ( ((pIds_.find(id) == pIds_.end()) ^ bInclude_)))
    return true;
  else
    return false;
}

void GenJetParticleSelector::init(const edm::EventSetup& es) {
  for(vpdt::iterator i = pdtList_.begin(); i != pdtList_.end(); ++i )
    i->setup(es);
}

