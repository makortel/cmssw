// Modify the pixel packing to make 100micron pixels possible. d.k. 2/02
//
#include "DataFormats/SiPixelDetId/interface/PixelChannelIdentifier.h"


/*
// Extract from CMSIM manual (version Thu Jul 31 16:38:50 MET DST 1997)
// --------------------------------------------------------------------
// DIGI format for pixel
//
// For pixel digitization one word per fired pixel is used. 
// The information includes pixel row and column number, time
// and charge information with 7, 9, 4 and 12 bits for each as shown below. 
//
//  :DETD    :TRAK  :PXBD    4   #. no. of digitization elements
//   #. name     no. bits
//     :V          7             #. row number
//     :W          9             #. column number
//     :TIME       4             #. time (ns)
//     :ADC       12             #. charge
//
// MODIFY 19.02.2002 for ORCA_6
// Change to enable 100micron row pixels, we than have 160 pixels in the v 
// direction.
//   #. name     no. bits
//     :V          8             #. row number        (256)
//     :W          9             #. column number     (512)
//     :TIME       4             #. time (ns)         (16)
//     :ADC       11             #. charge            (2048)
//
// MODIFY 20.5.2008 for SLHC Simulations
// Change to enable more pixels per row/column for larger modules in x and y
// Since time is not used in CMSSW anymore it is reduced to 1 bit and charge
// in adc counts is reduced since 255 is its maximum value
//   #. name     no. bits
//     :V          11            #. row number        (2048)
//     :W          11            #. column number     (2048)
//     :TIME       1             #. time (ns)         (2)
//     :ADC        9             #. charge            (512)

*/

// Initialization of static data members - DEFINES DIGI PACKING !
const PixelChannelIdentifier::Packing PixelChannelIdentifier::thePacking( 11, 11, 1, 9); // row, col, time, adc
