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
	actViewAsset = 0,
	actViewPlant = 1,
	actViewTurf = 2
};

enum GlyphTyp {
	gtNone = 0,
	gtPot = 1,
	gtBush = 2,
	gtClover = 33,
	gtPoapratensis = 34,
	gtHaircap = 35,
	gtHypericum = 36,
	gtSplineSprite = 65,
	gtBendTwistRollVariant = 97,
	gtImportGeom = 129,
};

enum GlyphGroup {
	ggGround = 0,
	ggGrass = 1,
	ggSprite = 2,
	ggVariant = 3,
	ggFile = 4
};

enum DisplayStat {
	dsTriangle = 256,
	dsDop = 257,
	dsPoint = 258,
	dsVoxel = 259
};

#define NumGlyphGroups 5

/// begin, end, 32 per group
static const int GlyphRange[NumGlyphGroups][2] = {
{1, 3},
{33, 37},
{65, 66},
{97, 98},
{129, 130}
};

static const char * PieceMimeStr = "image/x-garden-piece";

static inline int ToGroupType(int x) {
	return x>>5;
}

static const char sdecihexchart[16] = {'0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f'};

static inline void GenGlyphName(char* b)
{
	const int nid = rand();
	b[16] = '\0';
	
	for (int z = 0; z < 16; z++) {
        b[15-z] = sdecihexchart[(nid>>(z<<1)) & 15];
    }
}

/// up to 1024 instances of glyph with 1024 geom per glyph
static inline int GlyphTypeToGeomIdGroup(int gt)
{ return gt<<20; }

static inline int GeomIdToGlyphType(int gi)
{ return gi>>20; }

static inline int GeomIdInGlyphGroup(int gi)
{ return gi & 1048575; }

}
#endif