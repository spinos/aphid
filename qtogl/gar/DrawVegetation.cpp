/*
 *  DrawVegetation.cpp
 *  
 *
 *  Created by jian zhang on 4/19/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "DrawVegetation.h"
#include <geom/ATriangleMesh.h>
#include "PlantPiece.h"
#include "VegetationPatch.h"
#include <gl_heads.h>

using namespace aphid;

DrawVegetation::DrawVegetation()
{}

DrawVegetation::~DrawVegetation()
{}

void DrawVegetation::drawPointPatch(const VegetationPatch * vgp)
{
	if(vgp->numPlants() < 1) {
		return;
	}
	
	glPushMatrix();
	glMultMatrixf(vgp->transformationV());
	
	vgp->drawPoints();
	
	glPopMatrix();
	
}

void DrawVegetation::drawNaive(const VegetationPatch * vgp)
{
	if(vgp->numPlants() < 1) {
		return;
	}
	
	glPushMatrix();
	glMultMatrixf(vgp->transformationV());
	
	const int n = vgp->numPlants();
	for(int i=0;i<n;++i) {
		const PlantPiece * pl = vgp->plant(i);
		drawPlant(pl);
	}
	glPopMatrix();
}

void DrawVegetation::drawPlantPatch(const VegetationPatch * vgp)
{
	if(vgp->numPlants() < 1) {
		return;
	}
	
	glPushMatrix();
	glMultMatrixf(vgp->transformationV());
	
	glNormalPointer(GL_FLOAT, 0, (const GLfloat*)vgp->triNormalBuf() );
	glColorPointer(3, GL_FLOAT, 0, (const GLfloat*)vgp->triColorBuf() );
	glVertexPointer(3, GL_FLOAT, 0, (const GLfloat*)vgp->triPositionBuf() );
	glDrawArrays(GL_TRIANGLES, 0, vgp->triBufLength() );
	
	glPopMatrix();
	
}

void DrawVegetation::drawPlant(const PlantPiece * pl)
{
	drawPiece(pl);
}

void DrawVegetation::drawPiece(const PlantPiece * pl)
{
	glPushMatrix();
	float transbuf[16];
	const Matrix44F & tm = pl->transformMatrix();
	tm.glMatrix(transbuf);
	glMultMatrixf((const GLfloat*)transbuf);
	
	const ATriangleMesh * geo = pl->geometry();
	drawMesh(geo);
	
	const int n = pl->numBranches();
	for(int i=0;i<n;++i) {
		drawPiece(pl->branch(i) );
	}
	glPopMatrix();
}

void DrawVegetation::drawMesh(const ATriangleMesh * geo)
{
	if(!geo) {
		return;
	}
	
	glColorPointer(3, GL_FLOAT, 0, (const GLfloat*)geo->vertexColors() );
	glNormalPointer(GL_FLOAT, 0, (const GLfloat*)geo->vertexNormals() );
	glVertexPointer(3, GL_FLOAT, 0, (const GLfloat*)geo->points() );
	glDrawElements(GL_TRIANGLES, geo->numIndices(), GL_UNSIGNED_INT, geo->indices() );
}

void DrawVegetation::drawDopPatch(const VegetationPatch * vgp)
{
	glPushMatrix();
	glMultMatrixf(vgp->transformationV());
	
	vgp->drawASolidDop();
	glPopMatrix();

}

void DrawVegetation::drawVoxelPatch(const VegetationPatch * vgp)
{
	glPushMatrix();
	glMultMatrixf(vgp->transformationV());
	
	vgp->drawSolidGrid();
	glPopMatrix();

}	

void DrawVegetation::begin()
{
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
}

void DrawVegetation::end()
{
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
}
