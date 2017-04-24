/*
 *  ground.h
 *  garden
 *
 *  Created by jian zhang on 3/30/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */
 
#ifndef GAR_GROUND_H
#define GAR_GROUND_H

namespace gar {

static const char * GroundTypeNames[2] = {
"unknown",
"Flower Pot"
};

static const char * GroundTypeImages[2] = {
"unknown",
":/icons/unknown.png"
};

static const char * GroundTypeIcons[2] = {
":/icons/unknown.png",
":/icons/flowerpot.png"
};

static const char * GroundTypeDescs[2] = {
"unknown",
"flower pot"
};

static inline int ToGroundType(int x) {
	return x;
}

static const int GroundInPortRange[2][2] = {
{0,0},
{0,1}
};

static const char * GroundInPortRangeNames[2] = {
"inStem",
""
};

static const int GroundOutPortRange[2][2] = {
{0,0},
{0,0}
};

static const char * GroundOutPortRangeNames[2] = {
"",
""
};

}
#endif