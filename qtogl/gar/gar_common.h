/*
 *  gar_common.h
 *  
 *
 *  Created by jian zhang on 3/22/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_COMMON_H
#define GAR_COMMON_H

namespace gar {

enum ToolAction {
	actViewGraph = 0,
	actViewPlant = 1,
	actViewTurf = 2
};

enum GlyphTyp {
	gtNone = 0,
	gtPot = 1,
	gtClover = 33,
	gtPoapratensis = 34,
	gtHaircap = 35
};

enum GlyphGroup {
	ggGround = 0,
	ggGrass = 1
};

enum DisplayStat {
	dsTriangle = 256,
	dsDop = 257,
	dsPoint = 258,
	dsVoxel = 259
};

/// begin, end, 32 per group
static const int GlyphRange[2][2] = {
{1, 2},
{33, 36}
};

static const char * PieceMimeStr = "image/x-garden-piece";

static inline int ToGroupType(int x) {
	return x>>5;
}

}
#endif