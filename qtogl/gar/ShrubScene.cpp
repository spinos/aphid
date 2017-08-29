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
#include "Vegetation.h"
#include <geom/ATriangleMesh.h>
#include <attr/PieceAttrib.h>

using namespace aphid;

ShrubScene::ShrubScene(Vegetation * vege, QObject *parent)
    : QGraphicsScene(parent), 
PlantAssemble(vege),
m_lastSelectedGlyph(NULL)
{}

ShrubScene::~ShrubScene()
{}

GardenGlyph * ShrubScene::getGround()
{	
/// search selected first
	foreach(GardenGlyph* its_, m_selectedGlyph) {
		GardenGlyph *g = its_;
		PieceAttrib* pa = g->attrib();
		if(pa->isGround() )
			return g;
	}
	
	foreach(QGraphicsItem *its_, items()) {
		
		if(its_->type() == GardenGlyph::Type) {
			GardenGlyph *g = (GardenGlyph*) its_;
			PieceAttrib* pa = g->attrib();
			if(pa->isGround() )
				return g;
		}
	}
	qDebug()<<"  ERROR cannot find ground to grow on";
	return NULL;
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
	
	gar::SelectProfile selprof;
	return attr->selectGeom(&selprof);
}
