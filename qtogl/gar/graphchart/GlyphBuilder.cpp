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
#include "data/file.h"
#include <attr/PotAttribs.h>
#include <attr/BushAttribs.h>
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
		case gar::ggFile:
			buildFile(dst, gtyp);
		break;
		default:
			;
	}
	PieceAttrib * attr = buildAttrib(gtyp, ggrp);	
	dst->setAttrib(attr);
	dst->setGlyphType(gtyp);
	dst->finalizeShape();
}

PieceAttrib* GlyphBuilder::buildAttrib(const int & gtyp,
			const int & ggrp)
{
	switch (ggrp) {
		case gar::ggGround:
			return buildGroundAttrib(gtyp);			
		break;
		default:
			;
	}
	return (new PieceAttrib);
}

PieceAttrib* GlyphBuilder::buildGroundAttrib(const int & gtyp)
{
	if(gtyp == gar::gtPot)
		return (new PotAttribs);
		
	return (new BushAttribs);
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

void GlyphBuilder::buildFile(GardenGlyph * dst,
			const int & gtyp)
{
	const int gt = ToFileType(gtyp);
	const int & outBegin = FileOutPortRange[gt][0];
	const int & outEnd = FileOutPortRange[gt][1];
	for(int i=outBegin;i<outEnd;++i) {
		dst->addPort(QObject::tr(FileOutPortRangeNames[i]), true);
	}
}
