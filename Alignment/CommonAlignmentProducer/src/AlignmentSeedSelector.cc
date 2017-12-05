#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "Alignment/CommonAlignmentProducer/interface/AlignmentSeedSelector.h"

// constructor ----------------------------------------------------------------

AlignmentSeedSelector::AlignmentSeedSelector(const edm::ParameterSet & cfg) :
  applySeedNumber( cfg.getParameter<bool>( "applySeedNumber" ) ),
  minNSeeds ( cfg.getParameter<int>( "minNSeeds" ) ),
  maxNSeeds ( cfg.getParameter<int>( "maxNSeeds" ) )
{

  if (applySeedNumber)
	edm::LogInfo("AlignmentSeedSelector") 
	  << "apply seedNumber N<=" << minNSeeds;

}

// destructor -----------------------------------------------------------------

AlignmentSeedSelector::~AlignmentSeedSelector()
{}

void fillPSetDescription(edm::ParameterSetDescription& desc) {
  // default values are from the hat as there was no example where to get them
  desc.add<bool>("applySeedNumber", false);
  desc.add<int>("minNSeeds", 0);
  desc.add<int>("maxNSeeds", 0);
}


// do selection ---------------------------------------------------------------

AlignmentSeedSelector::Seeds 
AlignmentSeedSelector::select(const Seeds& seeds, const edm::Event& evt) const 
{
  Seeds result = seeds;

  // apply minimum/maximum multiplicity requirement (if selected)
  if (applySeedNumber) {
    if (result.size()<(unsigned int)minNSeeds || result.size()>(unsigned int)maxNSeeds ) result.clear();
  }

  return result;

}

// make basic cuts ------------------------------------------------------------

/* AlignmentSeedSelector::Seeds 
AlignmentSeedSelector::basicCuts(const Seeds& seeds) const 
{
  Seeds result;

  
  return result;
}

//-----------------------------------------------------------------------------

AlignmentSeedSelector::Seeds 
AlignmentSeedSelector::theNHighestPtSeeds(const Seeds& seeds) const
{
 
  Seeds result;


  return result;
}
*/
