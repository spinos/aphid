#ifndef BINNING_IMPLEMENT_H
#define BINNING_IMPLEMENT_H

#include "sahbvh_implement.h"

extern "C" {

void sahbvh_getNumBinningBlocks(uint * numBinningBlocks,
                        uint * numSpilledBlocks,
                        uint * totalBinningBlocks,
                        uint * totalSpilledBinningBlocks,
                        SplitId * splitIds,
                        EmissionBlock * emissionIds,
                        EmissionEvent * inEmissions,
                        int2 * rootRanges,
                        uint numClusters,
                        uint numEmissions);
}
#endif        //  #ifndef BINNING_IMPLEMENT_H

