/*
 *  GlyphBuilder.cpp
 *  
 *
 *  Created by jian zhang on 4/14/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "GlyphBuilder.h"
#include "GardenGlyph.h"
#include "gar_common.h"
#include "data/ground.h"
#include "data/grass.h"
#include <QString>

using namespace gar;

GlyphBuilder::GlyphBuilder()
{}

GlyphBuilder::~GlyphBuilder()
{}

void GlyphBuilder::build(GardenGlyph * dst,
			const int & gtyp,
			const int & ggrp)
{
	switch (ggrp) {
		case gar::ggGround:
			buildGround(dst, gtyp);
		break;
		case gar::ggGrass:
			buildGrass(dst, gtyp);
		break;
		default:
		;
	}
	dst->setGlyphType(gtyp);
	dst->finalizeShape();
}

void GlyphBuilder::buildGround(GardenGlyph * dst,
			const int & gtyp)
{
	const int gt = ToGroundType(gtyp);
	const int & inBegin = GroundInPortRange[gt][0];
	const int & inEnd = GroundInPortRange[gt][1];
	for(int i=inBegin;i<inEnd;++i) {
		dst->addPort(QObject::tr(GroundInPortRangeNames[i]), false);
	}
}

void GlyphBuilder::buildGrass(GardenGlyph * dst,
			const int & gtyp)
{
	const int gt = ToGrassType(gtyp);
	const int & inBegin = GrassInPortRange[gt][0];
	const int & inEnd = GrassInPortRange[gt][1];
	for(int i=inBegin;i<inEnd;++i) {
		dst->addPort(QObject::tr(GrassInPortRangeNames[i]), false);
	}
	const int & outBegin = GrassOutPortRange[gt][0];
	const int & outEnd = GrassOutPortRange[gt][1];
	for(int i=outBegin;i<outEnd;++i) {
		dst->addPort(QObject::tr(GrassOutPortRangeNames[i]), true);
	}
}
