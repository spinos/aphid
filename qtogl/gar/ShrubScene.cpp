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
#include "GardenGlyph.h"
#include "GlyphPort.h"
#include "GlyphConnection.h"
#include "gar_common.h"
#include "PlantPiece.h"
#include "VegetationPatch.h"
#include "data/ground.h"
#include "data/grass.h"
#include "data/clover.h"
#include <geom/ATriangleMesh.h>

using namespace gar;
using namespace aphid;

using namespace gar;

ShrubScene::ShrubScene(QObject *parent)
    : QGraphicsScene(parent)
{}

ShrubScene::~ShrubScene()
{ clearCachedGeom(); }

void ShrubScene::genPlants(VegetationPatch * vege)
{
	vege->clearPlants();
	
	for(int i=0;i<2000;++i) {
	
		if(vege->isFull() ) {
			break;
		}
		
		genAPlant(vege);
		
	}
	
	qDebug()<<" patch n "<<vege->numPlants();
}	

void ShrubScene::genAPlant(VegetationPatch * vege)
{	
	foreach(QGraphicsItem *its_, items()) {
		
		if(its_->type() == GardenGlyph::Type) {
			GardenGlyph *g = (GardenGlyph*) its_;
		
			if(g->glyphType() == gtPot) {
			
				PlantPiece * pl = new PlantPiece;
				assemblePlant(pl, g );
				if(!vege->addPlant(pl ) ) {
					delete pl;
				}
			}
		}
	}
	
}

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
		
	switch (gg) {
		case gar::ggGrass:
			PlantPiece * pl1 = new PlantPiece(pl);
			addGrassBranch(pl1, gl);
		break;
		default:
		;
	}
	
}

void ShrubScene::addGrassBranch(PlantPiece * pl, GardenGlyph * gl)
{
	const int gt = gar::ToGrassType(gl->glyphType() );
	const int ngeom = GrassGeomDeviations[gt];
	const int r = rand() % ngeom;
	
	int np = 0, nt = 0;
	const int * triind;
	const float * vertpos;
	const float * vertnml;
	float exclR;
	switch (gl->glyphType() ) {
		case gar::gtClover:
			np = sCloverNumVertices;
			nt = sCloverNumTriangleIndices / 3;
			triind = sCloverMeshTriangleIndices;
			vertpos = sCloverMeshVertices[r];
			vertnml = sCloverMeshNormals[r];
			exclR = sCloverExclRadius[r];
		break;
		default:
		;
	}
	
	if(np < 3) {
		return;
	}
	
	ATriangleMesh * msh = NULL;
	
	const int kgeom = (gl->glyphType()<<4) | r;
	if(m_cachedGeom.find(kgeom) != m_cachedGeom.end() ) {
		msh = m_cachedGeom[kgeom];
	}
	
	if(!msh) {
		msh = new ATriangleMesh;
		msh->create(np, nt);
		unsigned * indDst = msh->indices();
		memcpy(indDst, sCloverMeshTriangleIndices, nt * 12);
		Vector3F * pntDst = msh->points();
		memcpy(pntDst, vertpos, np * 12);
		Vector3F * nmlDst = msh->vertexNormals();
		memcpy(nmlDst, vertnml, np * 12);
		
		m_cachedGeom[kgeom] = msh;
	}
	
	pl->setGeometry(msh);
	pl->setExclR(exclR);
}

void ShrubScene::clearCachedGeom()
{
	std::map<int, aphid::ATriangleMesh * >::iterator it = m_cachedGeom.begin();
	for(;it!=m_cachedGeom.end();++it) {
		delete it->second;
	}
	m_cachedGeom.clear();
	
}
