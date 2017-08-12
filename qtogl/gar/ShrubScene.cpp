/*
 *  ShrubScene.cpp
 *  garden
 *
 *  Created by jian zhang on 3/30/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include <QtGui>
#include "ShrubScene.h"
#include "graphchart/GardenGlyph.h"
#include <qt/GlyphPort.h>
#include <qt/GlyphConnection.h>
#include "gar_common.h"
#include "PlantPiece.h"
#include "VegetationPatch.h"
#include "data/ground.h"
#include "data/grass.h"
#include "Vegetation.h"
#include <geom/ATriangleMesh.h>
#include "GrowthSample.h"
#include "attr/PieceAttrib.h"

using namespace aphid;

ShrubScene::ShrubScene(Vegetation * vege, QObject *parent)
    : QGraphicsScene(parent),
m_lastSelectedGlyph(NULL)
{ m_vege = vege; }

ShrubScene::~ShrubScene()
{}

void ShrubScene::assemblePlant(PlantPiece * pl, GardenGlyph * gl)
{
	foreach(QGraphicsItem *port_, gl->childItems()) {
		if (port_->type() != GlyphPort::Type) {
			continue;
		}

		GlyphPort *port = (GlyphPort*) port_;
		if(!port->isOutgoing() ) {
			addBranch(pl, port);
			
		}
	}
	pl->setExclRByChild();
	
}

void ShrubScene::addBranch(PlantPiece * pl, const GlyphPort * pt)
{
	const int n = pt->numConnections();
	if(n < 1) {
		return;
	}
	
	int r = rand() % n;
	
	const GlyphConnection * conn = pt->connection(r);
	const GlyphPort * srcpt = conn->port0();
	QGraphicsItem * top = srcpt->topLevelItem();
	if(top->type() != GardenGlyph::Type) {
		return;
	}
		
	GardenGlyph * gl = (GardenGlyph *)top;
	const int gg = gar::ToGroupType(gl->glyphType() );
		
	PlantPiece * pl1 = new PlantPiece(pl);
	switch (gg) {
		case gar::ggGrass:
		case gar::ggSprite:
		case gar::ggVariant:
			addGrassBranch(pl1, gl);
		break;
		default:
		;
	}
	
}

void ShrubScene::addGrassBranch(PlantPiece * pl, GardenGlyph * gl)
{
	PieceAttrib* attr = gl->attrib();
	const int ngeom = attr->numGeomVariations();
/// select randomly
	const int r = rand() % ngeom;
	float exclR = 1.f;
	ATriangleMesh * msh = attr->selectGeom(r, exclR);
	
	const int kgeom = gar::GlyphTypeToGeomIdGroup(gl->glyphType() ) | (gl->attribInstanceId() << 10) | r;
	
	if(!m_vege->findGeom(kgeom)) {
		m_vege->addGeom(kgeom, msh);
	}
	
	int geomInd = m_vege->getGeomInd(msh);
	pl->setGeometry(msh, geomInd);
	pl->setExclR(exclR);
}

void ShrubScene::genSinglePlant()
{
	GardenGlyph * gnd = getGround();
	if(!gnd) {
		m_vege->setNumPatches(0);
		return;
	}
	
	growOnGround(m_vege->patch(0), gnd);
	
	m_vege->setNumPatches(1);
	m_vege->voxelize();
}
	
void ShrubScene::genMultiPlant()
{
	GardenGlyph * gnd = getGround();
	if(!gnd) {
		m_vege->setNumPatches(0);
		return;
	}
	
	const int n = m_vege->getMaxNumPatches();
	for(int i=0;i<n;++i) {
		growOnGround(m_vege->patch(i), gnd);
	}
	m_vege->setNumPatches(n);
	m_vege->rearrange();
	m_vege->voxelize();
}

GardenGlyph * ShrubScene::getGround()
{
    GardenGlyph * firstGround = NULL;
	foreach(QGraphicsItem *its_, items()) {
		
		if(its_->type() == GardenGlyph::Type) {
			GardenGlyph *g = (GardenGlyph*) its_;
		
			if(g->glyphType() == gar::gtPot) {
			    if(m_selectedGlyph.contains(g) ) {
				std::cout<<"\n INFO grow by pot";
				return g;
				}
				if(!firstGround) {
				    firstGround = g;
				}
			}
			
			if(g->glyphType() == gar::gtBush) {
			    if(m_selectedGlyph.contains(g) ) {
				std::cout<<"\n INFO grow by bush";
				return g;
				}
				if(!firstGround) {
				    firstGround = g;
				}
			}
		}
	}
	if(!firstGround) {
	    std::cout<<"\n ERROR no ground to grow on";
	}
	return firstGround;
}

void ShrubScene::growOnGround(VegetationPatch * vege, GardenGlyph * gnd)
{
	vege->clearPlants();
	
	PieceAttrib * gndAttr = gnd->attrib();
	
	PlantPiece * pl = new PlantPiece;
	assemblePlant(pl, gnd );
	
	float zenithf = 0.1f;
	gar::Attrib* azenith = gndAttr->findAttrib(gar::nZenithNoise);
	if(azenith) {
	    azenith->getValue(zenithf);
	}
	float sizing = 1.f;
	gar::Attrib* asizing = gndAttr->findAttrib(gar::nGrowMargin);
	if(asizing) {
	    asizing->getValue(sizing);
	}
	
	float angleA = 0.f;
	gar::Attrib* aangle = gndAttr->findAttrib(gar::nGrowAngle);
	if(aangle) {
	    aangle->getValue(angleA);
	}
	
	float portion = 1.f;
	gar::Attrib* aportion = gndAttr->findAttrib(gar::nGrowPortion);
	if(aportion) {
	    aportion->getValue(portion);
	}
	
	GrowthSampleProfile prof;
	prof.m_numSampleLimit = 80;
	prof.m_sizing = pl->exclR() * sizing;
	prof.m_tilt = vege->tilt();
	prof.m_zenithNoise = zenithf;
	prof.m_spread = angleA;
	
	delete pl;
	
	GrowthSample gsmp;
	
	switch (gnd->glyphType()) {
		case gar::gtPot:
			prof.m_portion = .43f * portion;
			prof.m_angle = -1.f;
			gsmp.samplePot(prof);
			break;
		case gar::gtBush:
			prof.m_portion = .34f * portion;
			prof.m_angle = .41f;
			gsmp.sampleBush(prof);
			break;
		default:
			break;
	}

	const int& np = gsmp.numGrowthSamples();
	for(int i=0;i<np;++i) {
		pl = new PlantPiece;
		assemblePlant(pl, gnd );
		
		Matrix44F tm = gsmp.getGrowSpace(i, prof);
		pl->setTransformMatrix(tm);
		vege->addPlant(pl);
	}
	
}

void ShrubScene::selectGlyph(GardenGlyph* gl)
{
	if(!m_selectedGlyph.contains(gl) )
		m_selectedGlyph<<gl; 
	m_lastSelectedGlyph = gl;
}

void ShrubScene::deselectGlyph()
{
	foreach(GardenGlyph * gl, m_selectedGlyph) {
		gl->hideHalo();
	}
	m_selectedGlyph.clear();
	m_lastSelectedGlyph = NULL;
}

GardenGlyph* ShrubScene::lastSelectedGlyph()
{ return m_lastSelectedGlyph; }

const GardenGlyph* ShrubScene::lastSelectedGlyph() const
{ return m_lastSelectedGlyph; }

const ATriangleMesh* ShrubScene::lastSelectedGeom() const
{
	if(!m_lastSelectedGlyph) 
		return NULL;
	PieceAttrib* attr = m_lastSelectedGlyph->attrib();
	if(!attr->hasGeom())
		return NULL;
	float r;
/// first geom
	return attr->selectGeom(0, r);
}
