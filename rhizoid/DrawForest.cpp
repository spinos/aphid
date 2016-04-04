/*
 *  DrawForest.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 1/30/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "DrawForest.h"
#include <gl_heads.h>
#include <CircleCurve.h>
#include <DepthCull.h>
#include <ExampVox.h>

namespace aphid {

DrawForest::DrawForest() : m_showVoxLodThresold(1.f)
{
	m_scalbuf[0] = 1.f; 
	m_scalbuf[1] = 1.f; 
	m_scalbuf[2] = 1.f;
    m_wireColor[0] = m_wireColor[1] = m_wireColor[2] = 0.0675f;
}

DrawForest::~DrawForest() {}

void DrawForest::setScaleMuliplier(float x, float y, float z)
{ m_scalbuf[0] = x; m_scalbuf[1] = y; m_scalbuf[2] = z; }

void DrawForest::drawGround() 
{
	if(numActiveGroundFaces() < 1) return;
	sdb::Sequence<int> * prims = activeGround()->primIndices();
	const sdb::VectorArray<cvx::Triangle> & tris = triangles();
	
	glPushAttrib(GL_CURRENT_BIT);
	glDisable(GL_LIGHTING);
	glColor3f(.57f, .37f, 0.f);
	
	glBegin(GL_TRIANGLES);
	
	prims->begin();
	while(!prims->end() ) {
	
		const cvx::Triangle * t = tris[prims->key() ];
		drawFace(t->ind0(), t->ind1() );
		
		prims->next();
	}
	
	glEnd();
	glPopAttrib();
}

void DrawForest::drawFace(const int & geoId, const int & triId)
{
	const ATriangleMesh * mesh = groundMeshes()[geoId];
	Vector3F * p = mesh->points();
	unsigned * tri = mesh->triangleIndices(triId );
	glVertex3fv((GLfloat *)&p[tri[0]]);
	glVertex3fv((GLfloat *)&p[tri[1]]);
	glVertex3fv((GLfloat *)&p[tri[2]]);
		
}

void DrawForest::drawFaces(Geometry * geo, sdb::Sequence<unsigned> * components)
{
	ATriangleMesh * mesh = static_cast<ATriangleMesh *>(geo);
	Vector3F *p = mesh->points();
	components->begin();
	while(!components->end()) {
		unsigned * tri = mesh->triangleIndices(components->key() );
		glVertex3fv((GLfloat *)&p[tri[0]]);
		glVertex3fv((GLfloat *)&p[tri[1]]);
		glVertex3fv((GLfloat *)&p[tri[2]]);
		components->next();
	}
}

void DrawForest::drawWiredPlants()
{
	sdb::WorldGrid<sdb::Array<int, Plant>, Plant > * g = grid();
	if(g->isEmpty() ) return;
	const float margin = g->gridSize() * .13f;
	glDepthFunc(GL_LEQUAL);
	glPushAttrib(GL_LIGHTING_BIT | GL_CURRENT_BIT);
	glDisable(GL_LIGHTING);
	glColor3fv(m_wireColor);
	g->begin();
	while(!g->end() ) {
		BoundingBox cellBox = g->coordToGridBBox(g->key() );
        cellBox.expand(margin);
        if(!cullByFrustum(cellBox ) )
			drawWiredPlants(g->value() );
		g->next();
	}
	
	glPopAttrib();
}

void DrawForest::drawWiredPlants(sdb::Array<int, Plant> * cell)
{
	cell->begin();
	while(!cell->end() ) {
		drawWiredPlant(cell->value()->index);
		cell->next();
	}
}

void DrawForest::drawWiredPlant(PlantData * data)
{
	glPushMatrix();
    
	data->t1->glMatrix(m_transbuf);
	glMultMatrixf((const GLfloat*)m_transbuf);
	const ExampVox * v = plantExample(*data->t3);
	drawWireBox(v->geomCenterV(), v->geomScale() );
		
	glPopMatrix();
}

void DrawForest::drawPlants()
{
    sdb::WorldGrid<sdb::Array<int, Plant>, Plant > * g = grid();
	if(g->isEmpty() ) return;
	const float margin = g->gridSize() * .1f;
	glDepthFunc(GL_LEQUAL);
	glPushAttrib(GL_LIGHTING_BIT);
	glEnable(GL_LIGHTING);
	g->begin();
	while(!g->end() ) {
        BoundingBox cellBox = g->coordToGridBBox(g->key() );
        cellBox.expand(margin);
        if(!cullByFrustum(cellBox ) )
            drawPlants(g->value() );
		g->next();
	}
	
	glPopAttrib();
}

void DrawForest::drawPlants(sdb::Array<int, Plant> * cell)
{
	cell->begin();
	while(!cell->end() ) {
		drawPlant(cell->value()->index);
		cell->next();
	}
}

void DrawForest::drawPlant(PlantData * data)
{
	glPushMatrix();
    
	data->t1->glMatrix(m_transbuf);
	glMultMatrixf(m_transbuf);
	glScalef(m_scalbuf[0], m_scalbuf[1], m_scalbuf[2]);
	const ExampVox * v = plantExample(*data->t3);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, v->diffuseMaterialColor() );
	drawPlant(v , data);
	glPopMatrix();
}

void DrawForest::drawPlant(const ExampVox * v, PlantData * data)
{	
	if(m_showVoxLodThresold >.9999f) {
        drawSolidBox(v->geomCenterV(), v->geomScale() );
		return;
    }
	
	if(v->numBoxes() < 1) {
		drawSolidBox(v->geomCenterV(), v->geomScale() );
		return;
	}
	
	const Vector3F & localP = v->geomCenter();
	Vector3F worldP = data->t1->transform(localP);
	float r = v->geomExtent() * data->t1->getSide().length();
	if(cullByFrustum(worldP, r) ) {
		return;
	}
	
	float camZ = cameraDepth(worldP);
	float lod;
	if(cullByLod(camZ, r, m_showVoxLodThresold, 1.9f, lod) ) {
		drawSolidBox(v->geomCenterV(), v->geomScale() );
		return;
	}
	
	drawSolidBoxArray(v->boxPositionBuf(), v->boxNormalBuf(), 
						v->boxBufLength() );
}

void DrawForest::drawGridBounding()
{
	if(numPlants() < 1) return;
	drawBoundingBox(&gridBoundingBox() );
}

void DrawForest::drawGrid()
{
	sdb::WorldGrid<sdb::Array<int, Plant>, Plant > * g = grid();
	if(g->isEmpty() ) return;
	g->begin();
	while(!g->end() ) {
		drawBoundingBox(&g->coordToGridBBox(g->key() ) );
		g->next();
	}
}

void DrawForest::drawPlantBox(PlantData * data)
{
	glPushMatrix();
    
	data->t1->glMatrix(m_transbuf);
	glMultMatrixf((const GLfloat*)m_transbuf);
	const ExampVox * v = plantExample(*data->t3);
	drawWireBox(v->geomCenterV(), v->geomScale() );
		
	glPopMatrix();
}

void DrawForest::drawActivePlants()
{
	if(numActivePlants() < 1) return;
	glDepthFunc(GL_LEQUAL);
	glColor3f(.1f, .9f, .43f);
	sdb::Array<int, PlantInstance> * arr = activePlants();
	arr->begin();
	while(!arr->end() ) {
		drawPlantBox(arr->value()->m_reference->index );
		arr->next();
	}
}

void DrawForest::drawViewFrustum()
{
	const AFrustum & fr = frustum();
	glBegin(GL_LINES);
	for(int i=0; i < 4; i++) {
		glVertex3fv((const GLfloat*)fr.v(i) );
		glVertex3fv((const GLfloat*)fr.v(i+4) );
	}
	glEnd();
}

void DrawForest::drawBrush()
{
	glPushAttrib(GL_CURRENT_BIT);
	
    const float & radius = selectionRadius();
    const Vector3F & position = selectionCenter();
    const Vector3F & direction = selectionNormal();
    
	glPushMatrix();
    glTranslatef(position.x, position.y, position.z);
    glTranslatef(direction.x, direction.y, direction.z);
    
    m_useMat.setFrontOrientation(direction);
	m_useMat.scaleBy(radius);
    m_useMat.glMatrix(m_transbuf);
    
    draw3Circles(m_transbuf);
    glPopMatrix();
	
    glPopAttrib();
}

bool DrawForest::isVisibleInView(Plant * pl,
					const float lowLod, const float highLod)
{
	PlantData * d = pl->index;
	int typ = *d->t3;
	ExampVox * v = plantExample(typ);
	const Vector3F & localP = v->geomCenter();
	Vector3F worldP = d->t1->transform(localP);
	const float r = v->geomExtent() * d->t1->getSide().length();
	if(cullByFrustum(worldP, r) ) return false;
    
	float camZ;
	if(cullByDepth<cvx::Triangle, KdNTree<cvx::Triangle, KdNode4 > >(worldP, r * 2.f, camZ, ground() ) ) return false;
	
	if(lowLod > 0.f || highLod < 1.f) {
/// local z is negative
		camZ = -camZ;
		float lod;
		if(cullByLod(camZ, r, lowLod, highLod, lod ) ) return false;
	}
	return true;
}

void DrawForest::setShowVoxLodThresold(const float & x)
{ m_showVoxLodThresold = x; }

void DrawForest::setWireColor(const float & r, const float & g, const float & b)
{
    m_wireColor[0] = r;
    m_wireColor[1] = g;
    m_wireColor[2] = b;
}

}
//:~