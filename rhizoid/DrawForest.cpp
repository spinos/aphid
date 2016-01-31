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

DrawForest::DrawForest() 
{ m_defBox = BoundingBox(-1.f, -1.f, -1.f, 1.f, 1.f, 1.f); }

DrawForest::~DrawForest() {}

void DrawForest::drawGround() 
{
	glPushAttrib(GL_LIGHTING_BIT);
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
{ return (m_defBox.distance(0) + m_defBox.distance(2)) * .5f; }

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
	const GLfloat grayDiffuseMaterial[] = {0.47, 0.46, 0.45};
	// const GLfloat greenDiffuseMaterial[] = {0.33, 0.53, 0.37};
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
    
	float m[16];
	data->t1->glMatrix(m);
	glMultMatrixf((const GLfloat*)m);
	//glMultMatrixf(mScale);
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

void DrawForest::drawViewFrustum(const Matrix44F & cameraSpace, 
								const Matrix44F & worldInverseSpace,
								const float & h_fov, const float & aspectRatio)
{
	float fnear = -1.f;
	float ffar = -250000.f;
	float nearRight = fnear * h_fov;
	float nearLeft = -nearRight;
	float nearUp = nearRight * aspectRatio;
	float nearBottom = -nearUp;
	float farRight = ffar * h_fov;
	float farLeft = -farRight;
	float farUp = farRight * aspectRatio;
	float farBottom = -farUp;
	Vector3F clipNear[4];
	Vector3F clipFar[4];
	
	clipNear[0].set(nearLeft, nearBottom, fnear);
	clipNear[1].set(nearRight, nearBottom, fnear);
	clipNear[2].set(nearRight, nearUp, fnear);
	clipNear[3].set(nearLeft, nearUp, fnear);
	
	clipFar[0].set(farLeft, farBottom, ffar);
	clipFar[1].set(farRight, farBottom, ffar);
	clipFar[2].set(farRight, farUp, ffar);
	clipFar[3].set(farLeft, farUp, ffar);
	
	glPushMatrix();
    
	float m[16];
	cameraSpace.glMatrix(m);
	glMultMatrixf((const GLfloat*)m);
	
	glPushMatrix();
    worldInverseSpace.glMatrix(m);
	glMultMatrixf((const GLfloat*)m);
	
	glBegin(GL_LINES);
	for(int i=0; i < 4; i++) {
		glVertex3fv((const GLfloat*)&clipNear[i]);
		glVertex3fv((const GLfloat*)&clipFar[i]);
	}
	glEnd();
		
	glPopMatrix();
	glPopMatrix();

}
//:~