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

#define NUM_GROUND_PIECES 3

static const char * GroundTypeNames[NUM_GROUND_PIECES] = {
"unknown",
"Flower Pot",
"Bush"
};

static const char * GroundTypeImages[NUM_GROUND_PIECES] = {
"unknown",
":/icons/unknown.png",
":/icons/unknown.png"
};

static const char * GroundTypeIcons[NUM_GROUND_PIECES] = {
":/icons/unknown.png",
":/icons/evendist.png",
":/icons/bush.png"
};

static const char * GroundTypeDescs[NUM_GROUND_PIECES] = {
"unknown",
"plant within a circle and growing up",
"plant within a circle and growing with angles"
};

static inline int ToGroundType(int x) {
	return x;
}

static const int GroundInPortRange[NUM_GROUND_PIECES][2] = {
{0,0},
{0,1},
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