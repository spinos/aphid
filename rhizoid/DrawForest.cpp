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
#include <ExampVox.h>
#include <geom/ATriangleMesh.h>
#include <iostream>
#include <ogl/GlslInstancer.h>

namespace aphid {

DrawForest::DrawForest() : m_showVoxLodThresold(1.f),
m_enabled(true)
{
	m_wireColor[0] = m_wireColor[1] = m_wireColor[2] = 0.0675f;
}

DrawForest::~DrawForest() 
{}

void DrawForest::drawGround() 
{
	//std::cout<<" DrawForest draw ground begin"<<std::endl;
    if(!m_enabled) return;
	if(numActiveGroundFaces() < 1) return;
	sdb::Sequence<int> * prims = activeGround()->primIndices();
	const sdb::VectorArray<cvx::Triangle> & tris = triangles();
	
	glPushAttrib(GL_CURRENT_BIT);
	glDisable(GL_LIGHTING);
	glColor3f(.57f, .37f, 0.f);
	
	glBegin(GL_TRIANGLES);
	
	try {
	prims->begin();
	while(!prims->end() ) {
	
		const cvx::Triangle * t = tris[prims->key() ];
		drawFace(t->ind0(), t->ind1() );
		
		prims->next();
	}
	} catch (...) {
		std::cerr<<"DrawForest draw ground caught something";
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
    if(!m_enabled) return;
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
	if(!m_enabled) {
		return;
	}
	
	sdb::WorldGrid<sdb::Array<int, Plant>, Plant > * g = grid();
	if(g->isEmpty() ) {
		return;
	}
	
	const float margin = g->gridSize() * .13f;
	glDepthFunc(GL_LEQUAL);
	glPushAttrib(GL_LIGHTING_BIT | GL_CURRENT_BIT);
	glDisable(GL_LIGHTING);
	glColor3fv(m_wireColor);
	try {
	g->begin();
	while(!g->end() ) {
		BoundingBox cellBox = g->coordToGridBBox(g->key() );
        cellBox.expand(margin);
        if(!cullByFrustum(cellBox ) )
			drawWiredPlants(g->value() );
		g->next();
	}
	} catch (...) {
		std::cerr<<"DrawForest draw wired plants caught something";
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
	v->drawWiredBound();
	
	glPopMatrix();
}

void DrawForest::drawSolidPlants()
{
	//std::cout<<" DrawForest draw plants begin"<<std::endl;
    if(!m_enabled) {
		return;
	}
	
    sdb::WorldGrid<sdb::Array<int, Plant>, Plant > * g = grid();
	if(g->isEmpty() ) {
		return;
	}
	
	const float margin = g->gridSize() * .1f;
	
	glPushAttrib(GL_LIGHTING_BIT);
	
	Vector3F lightVec(1,1,1);
	lightVec = cameraSpace().transformAsNormal(lightVec);
	m_instancer->setDistantLightVec(lightVec);
	m_instancer->programBegin();
	
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	
	try {
	g->begin();
	while(!g->end() ) {
        BoundingBox cellBox = g->coordToGridBBox(g->key() );
		cellBox.expand(margin);
        if(!cullByFrustum(cellBox ) ) {
            drawPlantsInCell(g->value(), cellBox );
		}
		g->next();
	}
	} catch (...) {
		std::cerr<<"DrawForest draw plants caught something";
	}
	
	m_instancer->programEnd();
	
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	
	glPopAttrib();
}

void DrawForest::drawPlantsInCell(sdb::Array<int, Plant> * cell,
								const BoundingBox & box)
{	
	Vector3F worldP = box.center();
	const float r = gridSize();
	float camZ = cameraDepth(worldP);
	camZ += r;
	float lod;
	if(cullByLod(camZ, r*0.1f, .4f, 1.9f, lod) ) {
		drawPlantSolidBoundInCell(cell);
		return;
	}
	
	cell->begin();
	while(!cell->end() ) {
		drawPlant(cell->value()->index);
		cell->next();
	}
}

void DrawForest::drawPlantSolidBoundInCell(sdb::Array<int, Plant> * cell)
{
	cell->begin();
	while(!cell->end() ) {
		drawPlantSolidBound(cell->value()->index);

		cell->next();
	}
}

void DrawForest::drawPlantSolidBound(PlantData * data)
{
	const Matrix44F & trans = *(data->t1);
	glMultiTexCoord4f(GL_TEXTURE1, trans(0,0), trans(1,0), trans(2,0), trans(3,0) );
	glMultiTexCoord4f(GL_TEXTURE2, trans(0,1), trans(1,1), trans(2,1), trans(3,1) );
	glMultiTexCoord4f(GL_TEXTURE3, trans(0,2), trans(1,2), trans(2,2), trans(3,2) );
	    
	const ExampVox * v = plantExample(*data->t3);
	const float * c = v->diffuseMaterialColor();	
	m_instancer->setDiffueColorVec(c);
	v->drawSolidBound();
}

void DrawForest::drawPlant(PlantData * data)
{
	const Matrix44F & trans = *(data->t1);
	glMultiTexCoord4f(GL_TEXTURE1, trans(0,0), trans(1,0), trans(2,0), trans(3,0) );
	glMultiTexCoord4f(GL_TEXTURE2, trans(0,1), trans(1,1), trans(2,1), trans(3,1) );
	glMultiTexCoord4f(GL_TEXTURE3, trans(0,2), trans(1,2), trans(2,2), trans(3,2) );
	    
	const ExampVox * v = plantExample(*data->t3);
	const float * c = v->diffuseMaterialColor();
			
	m_instancer->setDiffueColorVec(c);
	drawPlant(v , data);
}

void DrawForest::drawPlant(const ExampVox * v, PlantData * data)
{	
	if(m_showVoxLodThresold >.9999f) {
        v->drawASolidDop();
		return;
    }
	
	const Vector3F & localP = v->geomCenter();
	Vector3F worldP = data->t1->transform(localP);
	float r = v->geomExtent() * data->t1->getSide().length();
	if(cullByFrustum(worldP, r) ) {
		return;
	}
	
	if(v->triBufLength() < 1) {
		v->drawASolidDop();
		return;
	}
	
	float camZ = cameraDepth(worldP);
	float lod;
	if(cullByLod(camZ, r, m_showVoxLodThresold, 1.9f, lod) ) {
		v->drawASolidDop();
	} else {
		v->drawSolidTriangles();
	}
}

void DrawForest::drawGridBounding()
{
	if(numPlants() < 1) return;
	drawBoundingBox(&gridBoundingBox() );
}

void DrawForest::drawGrid()
{
	std::cout<<" DrawForest draw grid begin"<<std::endl;
    if(!m_enabled) return;
	sdb::WorldGrid<sdb::Array<int, Plant>, Plant > * g = grid();
	if(g->isEmpty() ) return;
	try {
	g->begin();
	while(!g->end() ) {
		drawBoundingBox(&g->coordToGridBBox(g->key() ) );
		g->next();
	}
	} catch (...) {
		std::cerr<<"DrawForest draw grid caught something";
	}
}

void DrawForest::drawPlantBox(PlantData * data)
{
	glPushMatrix();
    
	data->t1->glMatrix(m_transbuf);
	glMultMatrixf((const GLfloat*)m_transbuf);
	const ExampVox * v = plantExample(*data->t3);
	//drawWireBox(v->geomCenterV(), v->geomScale() );
	v->drawWiredBound();
		
	glPopMatrix();
}

void DrawForest::drawActivePlants()
{
    if(!m_enabled) return;
	if(numActivePlants() < 1) return;
	glDepthFunc(GL_LEQUAL);
	glColor3f(.1f, .9f, .43f);
	sdb::Array<int, PlantInstance> * arr = activePlants();
	try {
	arr->begin();
	while(!arr->end() ) {
		drawPlantBox(arr->value()->m_reference->index );
		arr->next();
	}
	} catch (...) {
		std::cerr<<"DrawForest draw active plants caught something";
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
	
/// view-aligned circle
	glPushMatrix();
	
	Matrix44F vmat; 
	Vector3F s, u, f;
	s = cameraSpaceR()->getSide();
	s.normalize();
	s *= radius;
	
	u = cameraSpaceR()->getUp();
	u.normalize();
	u *= radius;
	
	f = cameraSpaceR()->getFront();
	f.normalize();
	f *= radius;
	
	vmat.setOrientations(s, u, f);
	vmat.setTranslation(position + direction);
	
    vmat.glMatrix(m_transbuf);
	
	drawZRing(m_transbuf);
	
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

void DrawForest::enableDrawing()
{
    std::cout<<"\n DrawForest enable draw";
    std::cout.flush();
    m_enabled = true;
}

void DrawForest::disableDrawing()
{
     std::cout<<"\n DrawForest disable draw";
     std::cout.flush();
     m_enabled = false;
}

}
//:~