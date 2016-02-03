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
{ drawDefFaceBuf(); }

void DrawForest::draw_a_box() const
{ drawDefBoundBuf(); }

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

Vector3F DrawForest::plantCenter(int idx) const
{ return m_defBox.center(); }

void DrawForest::calculateDefExtent()
{ m_boxExtent = m_defBox.radius(); }

float DrawForest::plantExtent(int idx) const
{ return m_boxExtent; }

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
	glColor3f(.1f, .9f, .43f);
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

bool DrawForest::isVisibleInView(sdb::Plant * pl,
					const float lowLod, const float highLod)
{
	sdb::PlantData * d = pl->index;
	int typ = *d->t3;
	Vector3F localP = plantCenter(typ);
	Vector3F worldP = d->t1->transform(localP);
	float r = plantSize(typ) * d->t1->getSide().length() * plantExtent(typ);
	if(cullByFrustum(worldP, r) ) return false;
	float camZ;
	if(cullByDepth(worldP, r, camZ) ) return false;
	if(lowLod > 0.f || highLod < 1.f) {
		if(cullByLod(camZ, r, lowLod, highLod ) ) return false;
	}
	return true;
}

void DrawForest::setDefBox(const float & a, 
					const float & b,
					const float & c,
					const float & d,
					const float & e,
					const float & f)
{
	m_defBox.m_data[0] = a;
	m_defBox.m_data[1] = b;
	m_defBox.m_data[2] = c;
	m_defBox.m_data[3] = d;
	m_defBox.m_data[4] = e;
	m_defBox.m_data[5] = f;
	calculateDefExtent();
	buildDefBoundBuf();
	buildDefFaceBuf();
}

void DrawForest::buildDefBoundBuf()
{
	Vector3F minb = m_defBox.getMin();
	Vector3F maxb = m_defBox.getMax();
	
	m_boxBoundBuf[0] = minb.x;
	m_boxBoundBuf[1] = minb.y;
	m_boxBoundBuf[2] = minb.z;
	m_boxBoundBuf[3] = maxb.x;
	m_boxBoundBuf[4] = minb.y;
	m_boxBoundBuf[5] = minb.z;
	m_boxBoundBuf[6] = minb.x;
	m_boxBoundBuf[7] = maxb.y;
	m_boxBoundBuf[8] = minb.z;
	m_boxBoundBuf[9] = maxb.x;
	m_boxBoundBuf[10] = maxb.y;
	m_boxBoundBuf[11] = minb.z;
	m_boxBoundBuf[12] = minb.x;
	m_boxBoundBuf[13] = minb.y;
	m_boxBoundBuf[14] = maxb.z;
	m_boxBoundBuf[15] = maxb.x;
	m_boxBoundBuf[16] = minb.y;
	m_boxBoundBuf[17] = maxb.z;
	m_boxBoundBuf[18] = minb.x;
	m_boxBoundBuf[19] = maxb.y;
	m_boxBoundBuf[20] = maxb.z;
	m_boxBoundBuf[21] = maxb.x;
	m_boxBoundBuf[22] = maxb.y;
	m_boxBoundBuf[23] = maxb.z;
		
	m_boxBoundBuf[24] = minb.x;
	m_boxBoundBuf[25] = minb.y;
	m_boxBoundBuf[26] = minb.z;
	m_boxBoundBuf[27] = minb.x;
	m_boxBoundBuf[28] = maxb.y;
	m_boxBoundBuf[29] = minb.z;
	m_boxBoundBuf[30] = maxb.x;
	m_boxBoundBuf[31] = minb.y;
	m_boxBoundBuf[32] = minb.z;
	m_boxBoundBuf[33] = maxb.x;
	m_boxBoundBuf[34] = maxb.y;
	m_boxBoundBuf[35] = minb.z;
	m_boxBoundBuf[36] = minb.x;
	m_boxBoundBuf[37] = minb.y;
	m_boxBoundBuf[38] = maxb.z;
	m_boxBoundBuf[39] = minb.x;
	m_boxBoundBuf[40] = maxb.y;
	m_boxBoundBuf[41] = maxb.z;
	m_boxBoundBuf[42] = maxb.x;
	m_boxBoundBuf[43] = minb.y;
	m_boxBoundBuf[44] = maxb.z;
	m_boxBoundBuf[45] = maxb.x;
	m_boxBoundBuf[46] = maxb.y;
	m_boxBoundBuf[47] = maxb.z;
		
	m_boxBoundBuf[48] = minb.x;
	m_boxBoundBuf[49] = minb.y;
	m_boxBoundBuf[50] = minb.z;
	m_boxBoundBuf[51] = minb.x;
	m_boxBoundBuf[52] = minb.y;
	m_boxBoundBuf[53] = maxb.z;
	m_boxBoundBuf[54] = maxb.x;
	m_boxBoundBuf[55] = minb.y;
	m_boxBoundBuf[56] = minb.z;
	m_boxBoundBuf[57] = maxb.x;
	m_boxBoundBuf[58] = minb.y;
	m_boxBoundBuf[59] = maxb.z;
	m_boxBoundBuf[60] = minb.x;
	m_boxBoundBuf[61] = maxb.y;
	m_boxBoundBuf[62] = minb.z;
	m_boxBoundBuf[63] = minb.x;
	m_boxBoundBuf[64] = maxb.y;
	m_boxBoundBuf[65] = maxb.z;
	m_boxBoundBuf[66] = maxb.x;
	m_boxBoundBuf[67] = maxb.y;
	m_boxBoundBuf[68] = minb.z;
	m_boxBoundBuf[69] = maxb.x;
	m_boxBoundBuf[70] = maxb.y;
	m_boxBoundBuf[71] = maxb.z;
		
}

void DrawForest::buildDefFaceBuf()
{
	Vector3F minb = m_defBox.getMin();
	Vector3F maxb = m_defBox.getMax();
	
	m_boxFaceBuf[0] = minb.x;
	m_boxFaceBuf[1] = minb.y;
	m_boxFaceBuf[2] = minb.z;
	m_boxFaceBuf[3] = minb.x;
	m_boxFaceBuf[4] = maxb.y;
	m_boxFaceBuf[5] = minb.z;
	m_boxFaceBuf[6] = maxb.x;
	m_boxFaceBuf[7] = maxb.y;
	m_boxFaceBuf[8] = minb.z;
	m_boxFaceBuf[9] = maxb.x;
	m_boxFaceBuf[10] = minb.y;
	m_boxFaceBuf[11] = minb.z;
	
	m_boxFaceBuf[12] = minb.x;
	m_boxFaceBuf[13] = minb.y;
	m_boxFaceBuf[14] = maxb.z;
	m_boxFaceBuf[15] = maxb.x;
	m_boxFaceBuf[16] = minb.y;
	m_boxFaceBuf[17] = maxb.z;
	m_boxFaceBuf[18] = maxb.x;
	m_boxFaceBuf[19] = maxb.y;
	m_boxFaceBuf[20] = maxb.z;
	m_boxFaceBuf[21] = minb.x;
	m_boxFaceBuf[22] = maxb.y;
	m_boxFaceBuf[23] = maxb.z;
	
	m_boxFaceBuf[24] = minb.x;
	m_boxFaceBuf[25] = minb.y;
	m_boxFaceBuf[26] = minb.z;
	m_boxFaceBuf[27] = minb.x;
	m_boxFaceBuf[28] = minb.y;
	m_boxFaceBuf[29] = maxb.z;
	m_boxFaceBuf[30] = minb.x;
	m_boxFaceBuf[31] = maxb.y;
	m_boxFaceBuf[32] = maxb.z;
	m_boxFaceBuf[33] = minb.x;
	m_boxFaceBuf[34] = maxb.y;
	m_boxFaceBuf[35] = minb.z;
	
	m_boxFaceBuf[36] = maxb.x;
	m_boxFaceBuf[37] = minb.y;
	m_boxFaceBuf[38] = minb.z;
	m_boxFaceBuf[39] = maxb.x;
	m_boxFaceBuf[40] = maxb.y;
	m_boxFaceBuf[41] = minb.z;
	m_boxFaceBuf[42] = maxb.x;
	m_boxFaceBuf[43] = maxb.y;
	m_boxFaceBuf[44] = maxb.z;
	m_boxFaceBuf[45] = maxb.x;
	m_boxFaceBuf[46] = minb.y;
	m_boxFaceBuf[47] = maxb.z;
	
	m_boxFaceBuf[48] = minb.x;
	m_boxFaceBuf[49] = minb.y;
	m_boxFaceBuf[50] = minb.z;
	m_boxFaceBuf[51] = maxb.x;
	m_boxFaceBuf[52] = minb.y;
	m_boxFaceBuf[53] = minb.z;
	m_boxFaceBuf[54] = maxb.x;
	m_boxFaceBuf[55] = minb.y;
	m_boxFaceBuf[56] = maxb.z;
	m_boxFaceBuf[57] = minb.x;
	m_boxFaceBuf[58] = minb.y;
	m_boxFaceBuf[59] = maxb.z;
	
	m_boxFaceBuf[60] = minb.x;
	m_boxFaceBuf[61] = maxb.y;
	m_boxFaceBuf[62] = minb.z;
	m_boxFaceBuf[63] = minb.x;
	m_boxFaceBuf[64] = maxb.y;
	m_boxFaceBuf[65] = maxb.z;
	m_boxFaceBuf[66] = maxb.x;
	m_boxFaceBuf[67] = maxb.y;
	m_boxFaceBuf[68] = maxb.z;
	m_boxFaceBuf[69] = maxb.x;
	m_boxFaceBuf[70] = maxb.y;
	m_boxFaceBuf[71] = minb.z;
}

void DrawForest::drawDefBoundBuf() const
{
	glBegin( GL_LINES );
	int i=0;
	for(;i<24;++i) glVertex3fv(&m_boxBoundBuf[i*3]);
	glEnd();
}

void DrawForest::drawDefFaceBuf() const
{
	glBegin(GL_TRIANGLES);
	glNormal3f(0.f, 0.f, -1.f);
	glVertex3fv(&m_boxFaceBuf[0]);
	glVertex3fv(&m_boxFaceBuf[3]);
	glVertex3fv(&m_boxFaceBuf[6]);
	glVertex3fv(&m_boxFaceBuf[6]);
	glVertex3fv(&m_boxFaceBuf[9]);
	glVertex3fv(&m_boxFaceBuf[0]);
	
	glNormal3f(0.f, 0.f, 1.f);
	glVertex3fv(&m_boxFaceBuf[12]);
	glVertex3fv(&m_boxFaceBuf[15]);
	glVertex3fv(&m_boxFaceBuf[18]);
	glVertex3fv(&m_boxFaceBuf[18]);
	glVertex3fv(&m_boxFaceBuf[21]);
	glVertex3fv(&m_boxFaceBuf[12]);
	
	glNormal3f(-1.f, 0.f, 0.f);
	glVertex3fv(&m_boxFaceBuf[24]);
	glVertex3fv(&m_boxFaceBuf[27]);
	glVertex3fv(&m_boxFaceBuf[30]);
	glVertex3fv(&m_boxFaceBuf[30]);
	glVertex3fv(&m_boxFaceBuf[33]);
	glVertex3fv(&m_boxFaceBuf[24]);
	
	glNormal3f(1.f, 0.f, 0.f);
	glVertex3fv(&m_boxFaceBuf[36]);
	glVertex3fv(&m_boxFaceBuf[39]);
	glVertex3fv(&m_boxFaceBuf[42]);
	glVertex3fv(&m_boxFaceBuf[42]);
	glVertex3fv(&m_boxFaceBuf[45]);
	glVertex3fv(&m_boxFaceBuf[36]);
	
	glNormal3f(0.f, -1.f, 0.f);
	glVertex3fv(&m_boxFaceBuf[48]);
	glVertex3fv(&m_boxFaceBuf[51]);
	glVertex3fv(&m_boxFaceBuf[54]);
	glVertex3fv(&m_boxFaceBuf[54]);
	glVertex3fv(&m_boxFaceBuf[57]);
	glVertex3fv(&m_boxFaceBuf[48]);
	
	glNormal3f(0.f, 1.f, 0.f);
	glVertex3fv(&m_boxFaceBuf[60]);
	glVertex3fv(&m_boxFaceBuf[63]);
	glVertex3fv(&m_boxFaceBuf[66]);
	glVertex3fv(&m_boxFaceBuf[66]);
	glVertex3fv(&m_boxFaceBuf[69]);
	glVertex3fv(&m_boxFaceBuf[60]);
	
	glEnd();
}
//:~