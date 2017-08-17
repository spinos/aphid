/*
 *  variation.h
 *  
 *
 *  Created by jian zhang on 3/30/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */
 
#ifndef GAR_VARIATION_H
#define GAR_VARIATION_H

namespace gar {
    
#define NUM_VARIATION_PIECES 4

static const char * VariationTypeNames[NUM_VARIATION_PIECES] = {
"unknown",
"BendTwistRoll",
"Directional",
"FoldCrumple",
};

static const char * VariationTypeImages[NUM_VARIATION_PIECES] = {
"unknown",
":/icons/bendtwistrollvariant.png",
":/icons/directional.png",
":/icons/aging.png",
};

static const char * VariationTypeIcons[NUM_VARIATION_PIECES] = {
":/icons/unknown.png",
":/icons/bendtwistrollvariant.png",
":/icons/directional.png",
":/icons/aging.png",
};

static const char * VariationTypeDescs[NUM_VARIATION_PIECES] = {
"unknown",
"one to many by Bend-Twist-Roll deformation",
"one to many by Directional Bending deformation",
"one to many by fold and crumple deformation",
};

static inline int ToVariationType(int x) {
	return x - 96;
}

static const int VariationInPortRange[NUM_VARIATION_PIECES][2] = {
{0,0},
{0,1},
{0,1},
{0,1},
};

static const char * VariationInPortRangeNames[2] = {
"inStem",
""
};

static const int VariationOutPortRange[NUM_VARIATION_PIECES][2] = {
{0,0},
{0,1},
{0,1},
{0,1},
};

static const char * VariationOutPortRangeNames[2] = {
"outStem",
""
};

static const int VariationGeomDeviations[NUM_VARIATION_PIECES] = {
0,
32,
36,
48,
};

}
#endif
