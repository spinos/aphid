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
#include <ExampVox.h>
#include <geom/ATriangleMesh.h>
#include "ForestCell.h"
#include "ForestGrid.h"
#include <ogl/GlslInstancer.h>
#include <ogl/RotationHandle.h>
#include <ogl/TranslationHandle.h>
#include <ogl/ScalingHandle.h>
#include <ogl/DrawSample.h>
#include "GrowOption.h"

namespace aphid {

DrawForest::DrawForest() : m_showVoxLodThresold(1.f),
m_enabled(true)
{
	m_wireColor[0] = m_wireColor[1] = m_wireColor[2] = 0.0675f;
	m_rotHand = new RotationHandle(&m_rotMat);
    m_transHand = new TranslationHandle(&m_rotMat);
	m_scalHand = new ScalingHandle(&m_rotMat);
}

DrawForest::~DrawForest() 
{
	delete m_rotHand;
    delete m_transHand;
	delete m_scalHand;
}

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
	
	sdb::WorldGrid<ForestCell, Plant > * g = grid();
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
        if(!cullByFrustum(cellBox ) ) {
			drawWiredPlants(g->value() );
		}
		
		g->next();
	}
	} catch (...) {
		std::cerr<<"DrawForest draw wired plants caught something";
	}
	glPopAttrib();
}

void DrawForest::drawWiredPlants(ForestCell * cell)
{
	int iExample = -1;
	ExampVox * v = 0;
	cell->begin();
	while(!cell->end() ) {
		if(cell->key().y != iExample) {
			iExample = cell->key().y;
			v = plantExample(iExample );
		}
		drawWiredPlant(cell->value()->index, v );
		
		cell->next();
	}
}

void DrawForest::drawWiredPlant(PlantData * data, 
						const ExampVox * v)
{
    if(!v) {
        std::cout<<"drawWiredPlant v is null";
        return;    
    }
    
	glPushMatrix();
    
	data->t1->glMatrix(m_transbuf);
	glMultMatrixf((const GLfloat*)m_transbuf);
	v->drawWiredBound();
	
	glPopMatrix();
}

void DrawForest::drawSolidPlants()
{
	//std::cout<<" DrawForest draw plants begin"<<std::endl;
    if(!m_enabled) {
		return;
	}
	
    sdb::WorldGrid<ForestCell, Plant > * g = grid();
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

void DrawForest::drawPlantsInCell(ForestCell * cell,
								const BoundingBox & box)
{
	if(!cell) {
		throw " drawPlantsInCell cell is null";
	}

	Vector3F worldP = box.center();
	const float r = gridSize();
	float camZ = cameraDepth(worldP);
	camZ += r;
	float lod;
	if(cullByLod(camZ, r * .2f, .99f, 1.9f, lod) ) {
		drawPlantSolidBoundInCell(cell);
		return;
	}
	
	int iExample = -1;
	ExampVox * v = 0;
	cell->begin();
	while(!cell->end() ) {
		if(cell->key().y != iExample) {
			iExample = cell->key().y;
			v = plantExample(iExample );
			if(v) {
			    const float * c = v->diffuseMaterialColor();
			    m_instancer->setDiffueColorVec(c);
			} else {
			    std::cout<<"drawWiredPlant v is null";
			}
			
		}
		
		if(v) {
		    drawPlant(cell->value()->index, v);
		}
		
		cell->next();
	}
}

void DrawForest::drawPlantSolidBoundInCell(ForestCell * cell)
{
	int iExample = -1;
	ExampVox * v = 0;
	cell->begin();
	while(!cell->end() ) {
		if(cell->key().y != iExample) {
			iExample = cell->key().y;
			v = plantExample(iExample );
			if(v) {
				const float * c = v->diffuseMaterialColor();	
				m_instancer->setDiffueColorVec(c);
			} else {
				std::cout<<"drawPlantSolidBoundInCell v is null";
			}
		}

		if(v) {
			drawPlantSolidBound(cell->value()->index, v);
		}

		cell->next();
	}
}

void DrawForest::drawPlantSolidBound(PlantData * data,
								const ExampVox * v)
{
	const Matrix44F & trans = *(data->t1);
	glMultiTexCoord4f(GL_TEXTURE1, trans(0,0), trans(1,0), trans(2,0), trans(3,0) );
	glMultiTexCoord4f(GL_TEXTURE2, trans(0,1), trans(1,1), trans(2,1), trans(3,1) );
	glMultiTexCoord4f(GL_TEXTURE3, trans(0,2), trans(1,2), trans(2,2), trans(3,2) );
	
	v->drawSolidBound();
}

void DrawForest::drawPlant(PlantData * data,
						const ExampVox * v)
{
	const Matrix44F & trans = *(data->t1);
	glMultiTexCoord4f(GL_TEXTURE1, trans(0,0), trans(1,0), trans(2,0), trans(3,0) );
	glMultiTexCoord4f(GL_TEXTURE2, trans(0,1), trans(1,1), trans(2,1), trans(3,1) );
	glMultiTexCoord4f(GL_TEXTURE3, trans(0,2), trans(1,2), trans(2,2), trans(3,2) );
	
	drawLODPlant(data, v);
}

void DrawForest::drawLODPlant(PlantData * data,
					const ExampVox * v)
{	
	if(m_showVoxLodThresold > .9999f) {
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
	if(!m_enabled) {
		return;
	}
	
	sdb::WorldGrid<ForestCell, Plant > * g = grid();
	if(g->isEmpty() ) {
		return;
	}
	
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

void DrawForest::drawSample()
{
	if(!m_enabled) {
		return;
	}
	
	sdb::WorldGrid<ForestCell, Plant > * g = grid();
	if(g->isEmpty() ) {
		return;
	}
	
	DrawSample::Profile drprof;
	//drprof.m_stride = sdb::SampleCache::DataStride;
	drprof.m_pointSize = 5.f;
	drprof.m_hasNormal = true;
	drprof.m_hasColor = false;
	
	DrawSample drs;
	drs.begin(drprof);
	
	try {
	g->begin();
	while(!g->end() ) {
		
		
		g->next();
	}
	} catch (...) {
		std::cerr<<"DrawForest draw sample caught something";
	}
	
	drs.end();
	
}

void DrawForest::drawActivePlants()
{
    if(!m_enabled) {
		return;
	}
	if(numActivePlants() < 1) {
		return;
	}
	
	glDepthFunc(GL_LEQUAL);
	glColor3f(.1f, .9f, .43f);
	PlantSelection::SelectionTyp * arr = activePlants();
	int iExample = -1;
	ExampVox * v = 0;
	try {
	arr->begin();
	while(!arr->end() ) {
		if(arr->key().y != iExample) {
			iExample = arr->key().y;
			v = plantExample(iExample );
		}
		drawWiredPlant(arr->value()->m_reference->index, v );
		
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
	glDepthFunc(GL_ALWAYS);
	
    const float & radius = selectionRadius();
    const Vector3F & position = selectionCenter();
    const Vector3F & direction = selectionNormal();
    m_useMat.setTranslation(position);
    m_useMat.setFrontOrientation(direction);
	m_useMat.scaleBy(radius);
	
/// view-aligned circle
	glPushMatrix();
	
	Matrix44F vmat; 
	Matrix33F rmat = cameraSpaceR()->rotation();
	rmat.orthoNormalize();
	rmat *= radius;
	
	vmat.setRotation(rmat);
	vmat.setTranslation(position);
	
    vmat.glMatrix(m_transbuf);
	
	drawZCircle(m_transbuf);
	
	glPopMatrix();
	glDepthFunc(GL_LEQUAL);
    glPopAttrib();
}

bool DrawForest::isVisibleInView(Plant * pl,
					const ExampVox * v,
					const float lowLod, const float highLod)
{
	PlantData * d = pl->index;
	const Vector3F & localP = v->geomCenter();
	Vector3F worldP = d->t1->transform(localP);
	const float r = v->geomExtent() * d->t1->getSide().length();
	if(cullByFrustum(worldP, r) ) {
		return false;
    }
	
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
    m_enabled = true;
}

void DrawForest::disableDrawing()
{
     std::cout<<"\n DrawForest disable draw";
     m_enabled = false;
}

void DrawForest::drawManipulator()
{
    if(manipulateMode() == manNone) {
        return;   
    }
	
	Matrix33F rmat = cameraSpaceR()->rotation();
	rmat.orthoNormalize();
	Matrix44F mat;
	mat.setRotation(rmat);
	const Vector3F & pos = selectionCenter();
	m_rotHand->setRadius(relativeSizeAtDepth(pos, .29f) );
    m_transHand->setRadius(relativeSizeAtDepth(pos, .29f) );
    m_scalHand->setRadius(relativeSizeAtDepth(pos, .29f) );
    
    switch (manipulateMode() ) {
    case manRotate:
        m_rotHand->draw(&mat);
        break;
    case manTranslate:
        m_transHand->draw(&mat);
        break;
	case manScaling:
        m_scalHand->draw(&mat);
        break;
    default:
        break;
    }
	
}

void DrawForest::updateManipulateSpace(GrowOption & option)
{
	const Vector3F & pos = selectionCenter();
	m_rotMat.setTranslation(pos);
	Matrix33F rot;
	if(option.m_alongNormal) {
		Vector3F u = selectionNormal();
		Vector3F f = u.perpendicular();
		Vector3F s = u.cross(f);
		rot.fill(s, u, f);
	}
	m_rotMat.setRotation(rot);
}

void DrawForest::startRotate(const Ray & r)
{
    disableDrawing();
    m_rotHand->begin(&r);
	calculateSelectedWeight();
    enableDrawing();
}

void DrawForest::processRotate(const Ray & r)
{
    disableDrawing();
    m_rotHand->rotate(&r);
    rotatePlant();
    enableDrawing();
}

void DrawForest::finishRotate()
{
    m_rotHand->end();
}

void DrawForest::getDeltaRotation(Matrix33F & mat,
					const float & weight) const
{
    m_rotHand->getDeltaRotation(mat, weight);
}

void DrawForest::startTranslate(const Ray & r)
{
    disableDrawing();
    m_transHand->begin(&r);
	calculateSelectedWeight();
    enableDrawing();
}
    
void DrawForest::processTranslate(const Ray & r)
{
    disableDrawing();
    m_transHand->translate(&r);
    translatePlant();
    enableDrawing();
}
    
void DrawForest::finishTranslate()
{ m_transHand->end(); }

void DrawForest::getDeltaTranslation(Vector3F & vec,
					const float & weight) const
{ m_transHand->getDeltaTranslation(vec, weight); }

void DrawForest::startResize(const Ray & r)
{
	disableDrawing();
    m_scalHand->begin(&r);
	calculateSelectedWeight();
    enableDrawing();
}

void DrawForest::processResize(const Ray & r)
{
	disableDrawing();
    m_scalHand->scale(&r);
    resizePlant();
    enableDrawing();
}

void DrawForest::getDeltaScaling(Vector3F & vec,
					const float & weight) const
{ m_scalHand->getDeltaScaling(vec, weight); }

void DrawForest::finishResize()
{ 
	m_scalHand->end(); 
	Matrix33F rot = m_rotMat.rotation();
	rot.orthoNormalize();
	m_rotMat.setRotation(rot);
}
	
}
//:~