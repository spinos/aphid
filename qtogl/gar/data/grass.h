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
    
#define NUM_GRASS_PIECES 5

static const char * GrassTypeNames[NUM_GRASS_PIECES] = {
"unknown",
"Clover",
"Poapratensis",
"Haircap",
"Hypericum"
};

static const char * GrassTypeImages[NUM_GRASS_PIECES] = {
"unknown",
":/images/clover.png",
":/images/poapratensis.png",
":/images/haircap.png",
":/images/hypericum.png"
};

static const char * GrassTypeIcons[NUM_GRASS_PIECES] = {
":/icons/unknown.png",
":/icons/clover.png",
":/icons/poapratensis.png",
":/icons/haircap.png",
":/icons/hypericum.png"
};

static const char * GrassTypeDescs[NUM_GRASS_PIECES] = {
"unknown",
"common clover \"three leaf\" \n16 deviations\nheight 4 unit",
"common \"meadow-grass\" \n16 deviations\nheight 8 unit",
"common hair moss \n16 deviations\nheight 4 unit",
"St. John's-worts \n16 deviations\nheight 14 unit"
};

static inline int ToGrassType(int x) {
	return x - 32;
}

static const int GrassInPortRange[NUM_GRASS_PIECES][2] = {
{0,0},
{0,0},
{0,0},
{0,0},
{0,0}
};

static const char * GrassInPortRangeNames[2] = {
"",
""
};

static const int GrassOutPortRange[NUM_GRASS_PIECES][2] = {
{0,0},
{0,1},
{0,1},
{0,1},
{0,1}
};

static const char * GrassOutPortRangeNames[2] = {
"outStem",
""
};

static const int GrassGeomDeviations[NUM_GRASS_PIECES] = {
0,
16,
16,
16,
16
};

}
#endif
