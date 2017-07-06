/*
 *  grass.h
 *  
 *
 *  Created by jian zhang on 3/30/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */
 
#ifndef GAR_GRASS_H
#define GAR_GRASS_H

namespace gar {

static const char * GrassTypeNames[3] = {
"unknown",
"Clover",
"Poapratensis"
};

static const char * GrassTypeImages[3] = {
"unknown",
":/images/clover.png",
":/images/poapratensis.png"
};

static const char * GrassTypeIcons[3] = {
":/icons/unknown.png",
":/icons/clover.png",
":/icons/poapratensis.png"
};

static const char * GrassTypeDescs[3] = {
"unknown",
"common clover \"three leaf\" \n16 deviations\nheight 4 unit",
"common \"meadow-grass\" \n16 deviations\nheight 8 unit"
};

static inline int ToGrassType(int x) {
	return x - 32;
}

static const int GrassInPortRange[3][2] = {
{0,0},
{0,0},
{0,0}
};

static const char * GrassInPortRangeNames[2] = {
"",
""
};

static const int GrassOutPortRange[3][2] = {
{0,0},
{0,1},
{0,1}
};

static const char * GrassOutPortRangeNames[2] = {
"outStem",
""
};

static const int GrassGeomDeviations[3] = {
0,
16,
16
};

}
#endif