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

static const char * GrassTypeNames[2] = {
"unknown",
"Clover"
};

static const char * GrassTypeImages[2] = {
"unknown",
":/images/clover.png"
};

static const char * GrassTypeIcons[2] = {
":/icons/unknown.png",
":/icons/clover.png"
};

static const char * GrassTypeDescs[2] = {
"unknown",
"common clover \"three leaf\" \n16 deviations\nheight 4 unit"
};

static inline int ToGrassType(int x) {
	return x - 32;
}

static const int GrassInPortRange[2][2] = {
{0,0},
{0,0}
};

static const char * GrassInPortRangeNames[2] = {
"",
""
};

static const int GrassOutPortRange[2][2] = {
{0,0},
{0,1}
};

static const char * GrassOutPortRangeNames[2] = {
"outStem",
""
};

}
#endif