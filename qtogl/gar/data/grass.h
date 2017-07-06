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

static const char * GrassTypeNames[4] = {
"unknown",
"Clover",
"Poapratensis",
"Haircap"
};

static const char * GrassTypeImages[4] = {
"unknown",
":/images/clover.png",
":/images/poapratensis.png",
":/images/haircap.png"
};

static const char * GrassTypeIcons[4] = {
":/icons/unknown.png",
":/icons/clover.png",
":/icons/poapratensis.png",
":/icons/haircap.png"
};

static const char * GrassTypeDescs[4] = {
"unknown",
"common clover \"three leaf\" \n16 deviations\nheight 4 unit",
"common \"meadow-grass\" \n16 deviations\nheight 8 unit",
"common hair moss \n16 deviations\nheight 4 unit",
};

static inline int ToGrassType(int x) {
	return x - 32;
}

static const int GrassInPortRange[4][2] = {
{0,0},
{0,0},
{0,0},
{0,0}
};

static const char * GrassInPortRangeNames[2] = {
"",
""
};

static const int GrassOutPortRange[4][2] = {
{0,0},
{0,1},
{0,1},
{0,1}
};

static const char * GrassOutPortRangeNames[2] = {
"outStem",
""
};

static const int GrassGeomDeviations[4] = {
0,
16,
16,
16
};

}
#endif