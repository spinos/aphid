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

DrawForest::DrawForest() 
{ 
	m_defBox = BoundingBox(-1.f, -1.f, -1.f, 1.f, 1.f, 1.f);
	m_scalbuf[0] = 1.f; 
	m_scalbuf[1] = 1.f; 
	m_scalbuf[2] = 1.f; 
    m_circle = new CircleCurve;
}

DrawForest::~DrawForest() {}

void DrawForest::setScaleMuliplier(float x, int idx)
{ m_scalbuf[idx] = x; }

void DrawForest::drawGround() 
{
	glPushAttrib(GL_CURRENT_BIT);
	glDisable(GL_LIGHTING);
	glColor3f(.57f, .37f, 0.f);
	
	glBegin(GL_TRIANGLES);
	SelectionContext * active = activeGround();
	std::map<Geometry *, sdb::Sequence<unsigned> * >::iterator it = active->geometryBegin();
	for(; it != active->geometryEnd(); ++it) {
		drawFaces(it->first, it->second);
	}
	glEnd();
	glPopAttrib();
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

BoundingBox * DrawForest::defBoxP()
{ return &m_defBox; }

const BoundingBox & DrawForest::defBox() const
{ return m_defBox; }

void DrawForest::draw_solid_box() const
{
	Vector3F minb = m_defBox.getMin();
	Vector3F maxb = m_defBox.getMax();
	
    glBegin(GL_QUADS);
	glNormal3f(0.f, 0.f, -1.f);
	glVertex3f(minb.x, minb.y, minb.z);
	glVertex3f(minb.x, maxb.y, minb.z);
	glVertex3f(maxb.x, maxb.y, minb.z);
	glVertex3f(maxb.x, minb.y, minb.z);
	
	glNormal3f(0.f, 0.f, 1.f);
	glVertex3f(minb.x, minb.y, maxb.z);
	glVertex3f(maxb.x, minb.y, maxb.z);
	glVertex3f(maxb.x, maxb.y, maxb.z);
	glVertex3f(minb.x, maxb.y, maxb.z);
	
	glNormal3f(-1.f, 0.f, 0.f);
	glVertex3f(minb.x, minb.y, minb.z);
	glVertex3f(minb.x, minb.y, maxb.z);
	glVertex3f(minb.x, maxb.y, maxb.z);
	glVertex3f(minb.x, maxb.y, minb.z);
	
	glNormal3f(1.f, 0.f, 0.f);
	glVertex3f(maxb.x, minb.y, minb.z);
	glVertex3f(maxb.x, maxb.y, minb.z);
	glVertex3f(maxb.x, maxb.y, maxb.z);
	glVertex3f(maxb.x, minb.y, maxb.z);
	
	glNormal3f(0.f, -1.f, 0.f);
	glVertex3f(minb.x, minb.y, minb.z);
	glVertex3f(maxb.x, minb.y, minb.z);
	glVertex3f(maxb.x, minb.y, maxb.z);
	glVertex3f(minb.x, minb.y, maxb.z);
	
	glNormal3f(0.f, 1.f, 0.f);
	glVertex3f(minb.x, maxb.y, minb.z);
	glVertex3f(minb.x, maxb.y, maxb.z);
	glVertex3f(maxb.x, maxb.y, maxb.z);
	glVertex3f(maxb.x, maxb.y, minb.z);
	glEnd();
}

void DrawForest::draw_a_box() const
{ drawBounding(m_defBox); }

void DrawForest::draw_coordsys() const
{
	Vector3F minb = m_defBox.getMin();
	Vector3F maxb = m_defBox.getMax();
	
	glBegin( GL_LINES );
	glColor3f(1.f, 0.f, 0.f);
			glVertex3f( 0.f, 0.f, 0.f );
			glVertex3f(maxb.x, 0.f, 0.f); 
	glColor3f(0.f, 1.f, 0.f);					
			glVertex3f( 0.f, 0.f, 0.f );
			glVertex3f(0.f, maxb.y, 0.f); 
	glColor3f(0.f, 0.f, 1.f);					
			glVertex3f( 0.f, 0.f, 0.f );
			glVertex3f(0.f, 0.f, maxb.z);		
	glEnd();
}

int DrawForest::activePlantId() const
{ return 0; }

float DrawForest::plantSize(int idx) const
{ 
	float a = m_defBox.distance(0) * .7f;
	float b = m_defBox.distance(2) * .7f;
	return a > b ? a : b ; 
}

void DrawForest::drawWiredPlants()
{
	glDepthFunc(GL_LEQUAL);
	
	sdb::WorldGrid<sdb::Array<int, sdb::Plant>, sdb::Plant > * g = grid();
	if(g->isEmpty() ) return;
	g->begin();
	while(!g->end() ) {
		drawWiredPlants(g->value() );
		g->next();
	}
}

void DrawForest::drawWiredPlants(sdb::Array<int, sdb::Plant> * cell)
{
	cell->begin();
	while(!cell->end() ) {
		drawWiredPlant(cell->value()->index);
		cell->next();
	}
}

void DrawForest::drawWiredPlant(sdb::PlantData * data)
{
	glPushMatrix();
    
	float m[16];
	data->t1->glMatrix(m);
	glMultMatrixf((const GLfloat*)m);
	//glMultMatrixf(mScale);
	draw_a_box();
		
	glPopMatrix();
}

void DrawForest::drawPlants()
{
	glDepthFunc(GL_LEQUAL);
	const GLfloat grayDiffuseMaterial[] = {0.47f, 0.46f, 0.45f};
	// const GLfloat greenDiffuseMaterial[] = {0.33f, 0.53f, 0.37f};
	glPushAttrib(GL_LIGHTING_BIT);
	glEnable(GL_LIGHTING);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, grayDiffuseMaterial);
		
	sdb::WorldGrid<sdb::Array<int, sdb::Plant>, sdb::Plant > * g = grid();
	if(g->isEmpty() ) return;
	g->begin();
	while(!g->end() ) {
		drawPlants(g->value() );
		g->next();
	}
	
	glDisable(GL_LIGHTING);
	glPopAttrib();
}

void DrawForest::drawPlants(sdb::Array<int, sdb::Plant> * cell)
{
	cell->begin();
	while(!cell->end() ) {
		drawPlant(cell->value()->index);
		cell->next();
	}
}

void DrawForest::drawPlant(sdb::PlantData * data)
{
	glPushMatrix();
    
	data->t1->glMatrix(m_transbuf);
	glMultMatrixf(m_transbuf);
	glScalef(m_scalbuf[0], m_scalbuf[1], m_scalbuf[2]);
	draw_solid_box();
		
	glPopMatrix();
}

void DrawForest::drawGridBounding()
{
	if(numPlants() < 1) return;
	drawBounding(gridBoundingBox() );
}

void DrawForest::drawGrid()
{
	sdb::WorldGrid<sdb::Array<int, sdb::Plant>, sdb::Plant > * g = grid();
	if(g->isEmpty() ) return;
	g->begin();
	while(!g->end() ) {
		drawBounding(g->coordToGridBBox(g->key() ) );
		g->next();
	}
}

void DrawForest::drawActivePlants()
{
	if(numActivePlants() < 1) return;
	glDepthFunc(GL_LEQUAL);
	glColor3f(.1f, 8.f, .3f);
	sdb::Array<int, sdb::PlantInstance> * arr = activePlants();
	arr->begin();
	while(!arr->end() ) {
		drawWiredPlant(arr->value()->m_reference->index );
		arr->next();
	}
}

void DrawForest::drawBounding(const BoundingBox & b) const
{
	Vector3F minb = b.getMin();
	Vector3F maxb = b.getMax();
	
	glBegin( GL_LINES );
	
	    glVertex3f(minb.x, minb.y, minb.z);
		glVertex3f(maxb.x, minb.y, minb.z);
		glVertex3f(minb.x, maxb.y, minb.z);
		glVertex3f(maxb.x, maxb.y, minb.z);
		glVertex3f(minb.x, minb.y, maxb.z);
		glVertex3f(maxb.x, minb.y, maxb.z);
		glVertex3f(minb.x, maxb.y, maxb.z);
		glVertex3f(maxb.x, maxb.y, maxb.z);
		
		glVertex3f(minb.x, minb.y, minb.z);
		glVertex3f(minb.x, maxb.y, minb.z);
		glVertex3f(maxb.x, minb.y, minb.z);
		glVertex3f(maxb.x, maxb.y, minb.z);
		glVertex3f(minb.x, minb.y, maxb.z);
		glVertex3f(minb.x, maxb.y, maxb.z);
		glVertex3f(maxb.x, minb.y, maxb.z);
		glVertex3f(maxb.x, maxb.y, maxb.z);
		
		glVertex3f(minb.x, minb.y, minb.z);
		glVertex3f(minb.x, minb.y, maxb.z);
		glVertex3f(maxb.x, minb.y, minb.z);
		glVertex3f(maxb.x, minb.y, maxb.z);
		glVertex3f(minb.x, maxb.y, minb.z);
		glVertex3f(minb.x, maxb.y, maxb.z);
		glVertex3f(maxb.x, maxb.y, minb.z);
		glVertex3f(maxb.x, maxb.y, maxb.z);
		
	glEnd();
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
    const float & radius = selectionRadius();
    const Vector3F & position = selectionCenter();
    const Vector3F & direction = selectionNormal();
    const float offset = radius * 0.05f;
    const float part = radius * 0.33f;
    glPushMatrix();
    glTranslatef(position.x, position.y, position.z);
    glTranslatef(direction.x * offset, direction.y * offset, direction.z * offset);
    
    glBegin(GL_LINES);
    glVertex3f(0.f, 0.f, 0.f);
    glVertex3f(direction.x * part, direction.y * part, direction.z * part);
    glEnd();
    
    glPushMatrix();
    glScalef(radius, radius, radius);
    glPushMatrix();
    
    m_useMat.setFrontOrientation(direction);
    m_useMat.glMatrix(m_transbuf);
    glMultMatrixf(m_transbuf);
	
    drawCircle();
    glPopMatrix();
    glPopMatrix();
    glPopMatrix();
    
}

void DrawForest::drawCircle() const
{
	Vector3F p;
	glBegin(GL_LINE_STRIP);
	for(unsigned i = 0; i < m_circle->numVertices(); i++) {
		p = m_circle->getCv(i);
		glVertex3f(p.x, p.y, p.z);
	}
	glEnd();
}

void DrawForest::drawDepthCull(double * localTm)
{
	DepthCull * culler = depthCuller();
	culler->setLocalSpace(localTm);
	culler->frameBufferBegin();
	culler->drawFrameBuffer(groundMeshes() );
	culler->frameBufferEnd();
	//culler->showFrameBuffer();
}
//:~