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
#include "data/poapratensis.h"
#include "data/haircap.h"
#include "Vegetation.h"
#include <geom/ATriangleMesh.h>

using namespace gar;
using namespace aphid;

using namespace gar;

ShrubScene::ShrubScene(Vegetation * vege, QObject *parent)
    : QGraphicsScene(parent)
{ m_vege = vege; }

ShrubScene::~ShrubScene()
{}

void ShrubScene::genPlants(VegetationPatch * vege)
{
	vege->clearPlants();
	
	for(int i=0;i<2000;++i) {
	
		if(vege->isFull() ) {
			break;
		}
		
		genAPlant(vege);
		
	}
	
	//qDebug()<<" patch n "<<vege->numPlants();
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
		
	PlantPiece * pl1 = new PlantPiece(pl);
	switch (gg) {
		case gar::ggGrass:
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
	const float * vertcol;
	const float * tritexcoord;
	float exclR;
	switch (gl->glyphType() ) {
		case gar::gtClover:
			np = sCloverNumVertices;
			nt = sCloverNumTriangleIndices / 3;
			triind = sCloverMeshTriangleIndices;
			vertpos = sCloverMeshVertices[r];
			vertnml = sCloverMeshNormals[r];
			exclR = sCloverExclRadius[r];
			vertcol = sCloverMeshVertexColors[r];
			tritexcoord = sCloverMeshTriangleTexcoords[r];
		break;
		case gar::gtPoapratensis:
			np = sPoapratensisNumVertices;
			nt = sPoapratensisNumTriangleIndices / 3;
			triind = sPoapratensisMeshTriangleIndices;
			vertpos = sPoapratensisMeshVertices[r];
			vertnml = sPoapratensisMeshNormals[r];
			exclR = sPoapratensisExclRadius[r];
			vertcol = sPoapratensisMeshVertexColors[r];
			tritexcoord = sPoapratensisMeshTriangleTexcoords[r];
		break;
		case gar::gtHaircap:
			np = sHaircapNumVertices;
			nt = sHaircapNumTriangleIndices / 3;
			triind = sHaircapMeshTriangleIndices;
			vertpos = sHaircapMeshVertices[r];
			vertnml = sHaircapMeshNormals[r];
			exclR = sHaircapExclRadius[r];
			vertcol = sHaircapMeshVertexColors[r];
			tritexcoord = sHaircapMeshTriangleTexcoords[r];
		break;
		default:
		;
	}
	
	if(np < 3) {
		return;
	}
	
	const int kgeom = (gl->glyphType()<<4) | r;
	ATriangleMesh * msh = m_vege->findGeom(kgeom);
	
	if(!msh) {
		msh = new ATriangleMesh;
		msh->create(np, nt);
		msh->createVertexColors(np);
		unsigned * indDst = msh->indices();
		memcpy(indDst, triind, nt * 12);
		Vector3F * pntDst = msh->points();
		memcpy(pntDst, vertpos, np * 12);
		Vector3F * nmlDst = msh->vertexNormals();
		memcpy(nmlDst, vertnml, np * 12);
		float * colDst = msh->vertexColors();
		memcpy(colDst, vertcol, np * 12);
		float * texcoordDst = msh->triangleTexcoords();
		memcpy(texcoordDst, tritexcoord, nt * 24);
		
		m_vege->addGeom(kgeom, msh);
	}
	
	int geomInd = m_vege->getGeomInd(msh);
	pl->setGeometry(msh, geomInd);
	pl->setExclR(exclR);
}

void ShrubScene::genSinglePlant()
{
	genPlants(m_vege->patch(0));
	m_vege->setNumPatches(1);
	m_vege->voxelize();
}
	
void ShrubScene::genMultiPlant()
{
	const int n = m_vege->getMaxNumPatches();
	for(int i=0;i<n;++i) {
		genPlants(m_vege->patch(i));
	}
	m_vege->setNumPatches(n);
	m_vege->rearrange();
	m_vege->voxelize();
}
