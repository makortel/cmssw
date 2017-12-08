#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/EventSetupInitTrait.h"

#include "RecoEgamma/ElectronIdentification/plugins/ElectronIDSelector.h"
#include "RecoEgamma/ElectronIdentification/plugins/ElectronIDSelectorCutBased.h"
#include "RecoEgamma/ElectronIdentification/plugins/ElectronIDSelectorNeuralNet.h"
#include "RecoEgamma/ElectronIdentification/plugins/ElectronIDSelectorLikelihood.h"
typedef ElectronIDSelector<ElectronIDSelectorCutBased>   EleIdCutBasedSel;
typedef ElectronIDSelector<ElectronIDSelectorNeuralNet>  EleIdNeuralNetSel;
typedef ElectronIDSelector<ElectronIDSelectorLikelihood> EleIdLikelihoodSel;
typedef ObjectSelector<
          EleIdCutBasedSel, 
          edm::RefVector<reco::GsfElectronCollection> 
         > EleIdCutBasedRef ;
typedef ObjectSelector<
          EleIdNeuralNetSel, 
          edm::RefVector<reco::GsfElectronCollection> 
         > EleIdNeuralNetRef ;
typedef ObjectSelector<
          EleIdLikelihoodSel, 
          edm::RefVector<reco::GsfElectronCollection> 
         > EleIdLikelihoodRef ;
template<>
void EleIdCutBasedRef::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // With the current structure it is impossible to create a fully
  // working fillDescriptions, so treat as if fillDescriptions would
  // not be implemented.
  ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
template<>
void EleIdNeuralNetRef::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // The configuration would be simple, but it would be impossible to
  // continue with the existing naming convention where the module
  // labels (eid*) and cfi files (electronId*) differ. The real
  // implementation is left for the domain experts.
  ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
template<>
void EleIdLikelihoodRef::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // The configuration would be simple, but it would be impossible to
  // continue with the existing naming convention where the module
  // labels (eid*) and cfi files (electronId*) differ. The real
  // implementation is left for the domain experts.
  ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
DEFINE_FWK_MODULE(EleIdCutBasedRef);
DEFINE_FWK_MODULE(EleIdNeuralNetRef);
DEFINE_FWK_MODULE(EleIdLikelihoodRef);

#include "RecoEgamma/ElectronIdentification/plugins/ElectronIDExternalProducer.h"
typedef ElectronIDExternalProducer<ElectronIDSelectorCutBased>   EleIdCutBasedExtProducer;
typedef ElectronIDExternalProducer<ElectronIDSelectorNeuralNet>  EleIdNeuralNetExtProducer;
typedef ElectronIDExternalProducer<ElectronIDSelectorLikelihood> EleIdLikelihoodExtProducer;
DEFINE_FWK_MODULE(EleIdCutBasedExtProducer);
DEFINE_FWK_MODULE(EleIdNeuralNetExtProducer);
DEFINE_FWK_MODULE(EleIdLikelihoodExtProducer);

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "RecoEgamma/ElectronIdentification/plugins/ElectronLikelihoodESSource.h"
DEFINE_FWK_EVENTSETUP_MODULE( ElectronLikelihoodESSource );

typedef ObjectSelector<EleIdCutBasedSel, reco::GsfElectronCollection> EleIdCutBased;
template<>
void EleIdCutBased::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // With the current structure it is impossible to create a fully
  // working fillDescriptions, so treat as if fillDescriptions would
  // not be implemented.
  ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
DEFINE_FWK_MODULE(EleIdCutBased);
