/*
 *  trunk.h
 *  
 *
 *  Created by jian zhang on 3/30/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */
 
#ifndef GAR_TRUNK_H
#define GAR_TRUNK_H

namespace gar {
    
#define NUM_TRUNK_PIECES 2

static const char * TrunkTypeNames[NUM_TRUNK_PIECES] = {
"unknown",
"Simple Trunk"
};

static const char * TrunkTypeImages[NUM_TRUNK_PIECES] = {
"unknown",
":/icons/unknown.png"
};

static const char * TrunkTypeIcons[NUM_TRUNK_PIECES] = {
":/icons/unknown.png",
":/icons/unknown.png"
};

static const char * TrunkTypeDescs[NUM_TRUNK_PIECES] = {
"unknown",
"for text purpose \n # deviations 1\n height unknown unit"
};

static inline int ToTrunkType(int x) {
	return x - 256;
}

static const int TrunkInPortRange[NUM_TRUNK_PIECES][2] = {
{0,0},
{0,0},
};

static const char * TrunkInPortRangeNames[2] = {
"inStem",
"inLeaf",
};

static const int TrunkOutPortRange[NUM_TRUNK_PIECES][2] = {
{0,0},
{0,1},
};

static const char * TrunkOutPortRangeNames[2] = {
"outStem",
""
};

static const int TrunkGeomDeviations[NUM_TRUNK_PIECES] = {
0,
1,
};

}
#endif
