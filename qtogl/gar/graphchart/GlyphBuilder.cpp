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
#include "data/billboard.h"
#include "data/variation.h"
#include "data/stem.h"
#include "data/twig.h"
#include <attr/PotAttribs.h>
#include <attr/BushAttribs.h>
#include <attr/ImportGeomAttribs.h>
#include <attr/SplineSpriteAttribs.h>
#include <attr/RibSpriteAttribs.h>
#include <attr/CloverProp.h>
#include <attr/PoapratensisProp.h>
#include <attr/HaircapProp.h>
#include <attr/HypericumProp.h>
#include <attr/BendTwistRollAttribs.h>
#include <attr/SplineCylinderAttribs.h>
#include <attr/DirectionalBendAttribs.h>
#include <attr/SimpleTwigAttribs.h>
#include <QString>
#include <iostream>

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
		case gar::ggSprite:
			buildSprite(dst, gtyp);
		break;
		case gar::ggVariant:
			buildVariant(dst, gtyp);
		break;
		case gar::ggStem:
			buildStem(dst, gtyp);
		break;
		case gar::ggTwig:
			buildTwig(dst, gtyp);
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
	PieceAttrib* res = NULL;
	switch (ggrp) {
		case gar::ggGround:
			res = buildGroundAttrib(gtyp);			
		break;
		case gar::ggFile:
			res = buildFileAttrib(gtyp);			
		break;
		case gar::ggSprite:
			res = buildSpriteAttrib(gtyp);
		break;
		case gar::ggVariant:
			res = buildVariantAttrib(gtyp);
		break;
		case gar::ggGrass:
			res = buildGrassAttrib(gtyp);
		break;
		case gar::ggStem:
			res = buildStemAttrib(gtyp);
		break;
		case gar::ggTwig:
			res = buildTwigAttrib(gtyp);
		break;
		default:
			res = new PieceAttrib;
	}
	return res;
}

PieceAttrib* GlyphBuilder::buildGrassAttrib(const int & gtyp)
{
	if(gtyp == gar::gtClover)
		return (new CloverProp);
	if(gtyp == gar::gtPoapratensis)
		return (new PoapratensisProp);
	if(gtyp == gar::gtHaircap)
		return (new HaircapProp);
	if(gtyp == gar::gtHypericum)
		return (new HypericumProp);
		
	return (new PieceAttrib);
}

PieceAttrib* GlyphBuilder::buildGroundAttrib(const int & gtyp)
{
	if(gtyp == gar::gtPot)
		return (new PotAttribs);
		
	return (new BushAttribs);
}

PieceAttrib* GlyphBuilder::buildFileAttrib(const int & gtyp)
{
	if(gtyp == gar::gtImportGeom)
		return (new ImportGeomAttribs);
		
	return (new PieceAttrib);
}

PieceAttrib* GlyphBuilder::buildSpriteAttrib(const int & gtyp)
{
	if(gtyp == gar::gtSplineSprite)
		return (new SplineSpriteAttribs);
		
	if(gtyp == gar::gtRibSprite)
		return (new RibSpriteAttribs);
		
	return (new PieceAttrib);
}

PieceAttrib* GlyphBuilder::buildVariantAttrib(const int & gtyp)
{
	if(gtyp == gar::gtBendTwistRollVariant)
		return (new BendTwistRollAttribs);
	
	if(gtyp == gar::gtDirectionalVariant)
		return (new DirectionalBendAttribs);
		
	return (new PieceAttrib);
}

PieceAttrib* GlyphBuilder::buildStemAttrib(const int & gtyp)
{
	if(gtyp == gtSplineCylinder)
		return (new SplineCylinderAttribs);
		
	return (new PieceAttrib);
}

PieceAttrib* GlyphBuilder::buildTwigAttrib(const int & gtyp)
{
	if(gtyp == gtSimpleTwig)
		return (new SimpleTwigAttribs);
		
	return (new PieceAttrib);
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

void GlyphBuilder::buildSprite(GardenGlyph * dst,
			const int & gtyp)
{
	const int gt = ToBillboardType(gtyp);
	const int & outBegin = BillboardOutPortRange[gt][0];
	const int & outEnd = BillboardOutPortRange[gt][1];
	for(int i=outBegin;i<outEnd;++i) {
		dst->addPort(QObject::tr(BillboardOutPortRangeNames[i]), true);
	}
}

void GlyphBuilder::buildVariant(GardenGlyph * dst,
			const int & gtyp)
{
	const int gt = ToVariationType(gtyp);
	const int & inBegin = VariationInPortRange[gt][0];
	const int & inEnd = VariationInPortRange[gt][1];
	for(int i=inBegin;i<inEnd;++i) {
		dst->addPort(QObject::tr(VariationInPortRangeNames[i]), false);
	}
	const int & outBegin = VariationOutPortRange[gt][0];
	const int & outEnd = VariationOutPortRange[gt][1];
	for(int i=outBegin;i<outEnd;++i) {
		dst->addPort(QObject::tr(VariationOutPortRangeNames[i]), true);
	}
}

void GlyphBuilder::buildStem(GardenGlyph * dst,
			const int & gtyp)
{
	const int gt = ToStemType(gtyp);
	const int & inBegin = StemInPortRange[gt][0];
	const int & inEnd = StemInPortRange[gt][1];
	for(int i=inBegin;i<inEnd;++i) {
		dst->addPort(QObject::tr(StemInPortRangeNames[i]), false);
	}
	const int & outBegin = StemOutPortRange[gt][0];
	const int & outEnd = StemOutPortRange[gt][1];
	for(int i=outBegin;i<outEnd;++i) {
		dst->addPort(QObject::tr(StemOutPortRangeNames[i]), true);
	}
}

void GlyphBuilder::buildTwig(GardenGlyph * dst,
			const int & gtyp)
{
	const int gt = ToTwigType(gtyp);
	const int & inBegin = TwigInPortRange[gt][0];
	const int & inEnd = TwigInPortRange[gt][1];
	for(int i=inBegin;i<inEnd;++i) {
		dst->addPort(QObject::tr(TwigInPortRangeNames[i]), false);
	}
	const int & outBegin = TwigOutPortRange[gt][0];
	const int & outEnd = TwigOutPortRange[gt][1];
	for(int i=outBegin;i<outEnd;++i) {
		dst->addPort(QObject::tr(TwigOutPortRangeNames[i]), true);
	}
}
	