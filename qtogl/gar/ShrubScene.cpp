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
#include "Vegetation.h"
#include <geom/ATriangleMesh.h>
#include "GrowthSample.h"
#include "data/clover.h"
#include "data/poapratensis.h"
#include "data/haircap.h"
#include "data/haircap.h"
#include "data/hypericum.h"

using namespace gar;
using namespace aphid;

using namespace gar;

ShrubScene::ShrubScene(Vegetation * vege, QObject *parent)
    : QGraphicsScene(parent)
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
		case gar::gtHypericum:
			np = sHypericumNumVertices;
			nt = sHypericumNumTriangleIndices / 3;
			triind = sHypericumMeshTriangleIndices;
			vertpos = sHypericumMeshVertices[r];
			vertnml = sHypericumMeshNormals[r];
			exclR = sHypericumExclRadius[r];
			vertcol = sHypericumMeshVertexColors[r];
			tritexcoord = sHypericumMeshTriangleTexcoords[r];
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
	foreach(QGraphicsItem *its_, items()) {
		
		if(its_->type() == GardenGlyph::Type) {
			GardenGlyph *g = (GardenGlyph*) its_;
		
			if(g->glyphType() == gtPot) {
			
				std::cout<<"\n INFO grow by pot";
				return g;
			}
			
			if(g->glyphType() == gtBush) {
			
				std::cout<<"\n INFO grow by bush";
				return g;
			}
		}
	}
	std::cout<<"\n ERROR no ground to grow on";
	return NULL;
}

void ShrubScene::growOnGround(VegetationPatch * vege, GardenGlyph * gnd)
{
	vege->clearPlants();
	
	PlantPiece * pl = new PlantPiece;
	assemblePlant(pl, gnd );
	
	GrowthSampleProfile prof;
	prof.m_numSampleLimit = 80;
	prof.m_sizing = pl->exclR();
	prof.m_tilt = vege->tilt();
	
	delete pl;
	
	GrowthSample gsmp;
	
	switch (gnd->glyphType()) {
		case gtPot:
			prof.m_portion = .43f;
			prof.m_angle = -1.f;
			gsmp.samplePot(prof);
			break;
		case gtBush:
			prof.m_portion = .34f;
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
