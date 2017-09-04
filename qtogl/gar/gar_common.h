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

static const float DEV_VERSION_MAJOR = 1.4;
static const int DEV_VERSION_MINOR = 3;

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
	gtRibSprite = 66,
	gtBladeSprite = 67,
	gtOvalSprite = 68,
	gtReniformSprite = 69,
	gtBendTwistRollVariant = 97,
	gtDirectionalVariant = 98,
	gtFoldCrumpleVariant = 99,
	gtBlockDeformVariant = 100,
	gtImportGeom = 129,
	gtSplineCylinder = 161,
	gtMonopodial = 162,
	gtSimpleTwig = 193,
	gtSimpleBranch = 225,
};

enum GlyphGroup {
	ggGround = 0,
	ggGrass = 1,
	ggSprite = 2,
	ggVariant = 3,
	ggFile = 4,
	ggStem = 5,
	ggTwig = 6,
	ggBranch = 7,
};

enum DisplayStat {
	dsTriangle = 256,
	dsDop = 257,
	dsPoint = 258,
	dsVoxel = 259
};

#define NumGlyphGroups 8

/// begin, end, 32 per group
static const int GlyphRange[NumGlyphGroups][2] = {
{1, 3},
{33, 37},
{65, 70},
{97, 101},
{129, 130},
{161, 163},
{193, 194},
{225, 226},
};

static const char * PieceMimeStr = "image/x-garden-piece";

static inline int ToGroupType(int x) {
	return x>>5;
}

static inline int ToGroupBegin(int x) {
	return x<<5;
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