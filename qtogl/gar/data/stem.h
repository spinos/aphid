/*
 *  stem.h
 *  
 *
 *  Created by jian zhang on 3/30/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */
 
#ifndef GAR_STEM_H
#define GAR_STEM_H

namespace gar {
    
#define NUM_STEM_PIECES 3

static const char * StemTypeNames[NUM_STEM_PIECES] = {
"unknown",
"Spline Cylinder",
"Monopodial Default",
};

static const char * StemTypeImages[NUM_STEM_PIECES] = {
"unknown",
":/icons/unknown.png",
":/icons/unknown.png",
};

static const char * StemTypeIcons[NUM_STEM_PIECES] = {
":/icons/unknown.png",
":/icons/stem.png",
":/icons/monopodial.png",
};

static const char * StemTypeDescs[NUM_STEM_PIECES] = {
"unknown",
"cylinder with radius and height spline \n 1 deviations\n height unknown unit",
"default monopodial unit 4 buds \n 1 deviations\n height unknown unit",
};

static inline int ToStemType(int x) {
	return x - 160;
}

static const int StemInPortRange[NUM_STEM_PIECES][2] = {
{0,0},
{0,0},
{0,0},
};

static const char * StemInPortRangeNames[2] = {
"",
""
};

static const int StemOutPortRange[NUM_STEM_PIECES][2] = {
{0,0},
{0,1},
{0,1},
};

static const char * StemOutPortRangeNames[2] = {
"outStem",
""
};

static const int StemGeomDeviations[NUM_STEM_PIECES] = {
0,
1,
1,
};

}
#endif
