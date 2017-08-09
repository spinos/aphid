/*
 *  billboard.h
 *  
 *
 *  Created by jian zhang on 3/30/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */
 
#ifndef GAR_BILLBOARD_H
#define GAR_BILLBOARD_H

namespace gar {
    
#define NUM_BILLBOARD_PIECES 2

static const char * BillboardTypeNames[NUM_BILLBOARD_PIECES] = {
"unknown",
"Spline Patch"
};

static const char * BillboardTypeImages[NUM_BILLBOARD_PIECES] = {
"unknown",
":/icons/splinesprite.png"
};

static const char * BillboardTypeIcons[NUM_BILLBOARD_PIECES] = {
":/icons/unknown.png",
":/icons/splinesprite.png"
};

static const char * BillboardTypeDescs[NUM_BILLBOARD_PIECES] = {
"unknown",
"segmented piece shaped by two splines"
};

static inline int ToBillboardType(int x) {
	return x - 64;
}

static const int BillboardInPortRange[NUM_BILLBOARD_PIECES][2] = {
{0,0},
{0,0},
};

static const char * BillboardInPortRangeNames[2] = {
"",
""
};

static const int BillboardOutPortRange[NUM_BILLBOARD_PIECES][2] = {
{0,0},
{0,1},
};

static const char * BillboardOutPortRangeNames[2] = {
"outStem",
""
};

static const int BillboardGeomDeviations[NUM_BILLBOARD_PIECES] = {
0,
1,
};

}
#endif
