#include "AdeniumWorld.h"
#include <BvhTriangleSystem.h>
#include <GeoDrawer.h>
#include <CudaBase.h>
#include <CUDABuffer.h>
#include <BvhBuilder.h>
#include <PerspectiveCamera.h>
#include <TriangleDifference.h>
#include <ATetrahedronMesh.h>
#include "AdeniumRender.h"
#include <WorldDbgDraw.h>
#include <tetrahedron_math.h>
#include <TriangleAnchorDeformer.h>
#include <OVelocityFile.h>

WorldDbgDraw * AdeniumWorld::DbgDrawer = 0;
GLuint AdeniumWorld::m_texture = 0;

AdeniumWorld::AdeniumWorld() :
m_numObjects(0),
m_difference(0),
m_deformedMesh(0),
m_enableRayCast(true),
m_tetraMesh(0)
{
    m_image = new AdeniumRender;
    m_tetraDeformer = new TriangleAnchorDeformer;
    m_velocityOut = new OVelocityFile;
    if(!m_velocityOut->create("./velocity.tmp"))
        std::cout<<"\n WARNING: adenium world cannot create velocity file in current dir!\n";
}

AdeniumWorld::~AdeniumWorld() 
{
    delete m_image;
    delete m_tetraDeformer;
    if(m_difference) delete m_difference;
    if(m_deformedMesh) delete m_deformedMesh;
    if(m_tetraMesh) delete m_tetraMesh;
    delete m_velocityOut;
}

void AdeniumWorld::setRestMesh(ATriangleMesh * m)
{
    const Vector3F t = m->boundingCenter();
    m_restSpaceInv.setIdentity();
    m_restSpaceInv.setTranslation(t);
    m_restSpaceInv.inverse();
    m->moveIntoSpace(m_restSpaceInv);

    if(m_difference) delete m_difference;
    m_difference = new TriangleDifference(m);
    m_tetraDeformer->setDifference(m_difference);
}

bool AdeniumWorld::matchRestMesh(ATriangleMesh * m)
{
    return m_difference->matchTarget(m);
}

void AdeniumWorld::addTriangleSystem(BvhTriangleSystem * tri)
{
    if(m_numObjects == 32) return;
    m_objects[m_numObjects] = tri;
    m_numObjects++;
}

void AdeniumWorld::addTetrahedronMesh(ATetrahedronMesh * tetra)
{ 
    m_tetraMesh = tetra;
    m_tetraMesh->moveIntoSpace(m_restSpaceInv);
    m_difference->requireQ(tetra);
    m_tetraDeformer->setMesh(m_tetraMesh);
}

void AdeniumWorld::setBvhBuilder(BvhBuilder * builder)
{ CudaLinearBvh::Builder = builder; }

void AdeniumWorld::draw(BaseCamera * camera)
{
    unsigned i = 0;
    for(;i<m_numObjects; i++)
        drawTriangle(m_objects[i]);
    drawTetrahedron();
    drawAnchors();
	drawOverallTranslation();
	dbgDraw();
}

void AdeniumWorld::dbgDraw()
{
	if(!DbgDrawer) return;
#if DRAW_BVH_HIERARCHY
    unsigned i;
    for(i=0; i< m_numObjects; i++) {
        DbgDrawer->showBvhHierarchy(m_objects[i]);
	}
#endif
}

void AdeniumWorld::drawTriangle(TriangleSystem * tri)
{
    glEnable(GL_DEPTH_TEST);
	glColor3f(0.6f, 0.62f, 0.6f);
    
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	
    glEnableClientState(GL_VERTEX_ARRAY);

	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)tri->hostX());
	glDrawElements(GL_TRIANGLES, tri->numTriangleFaceVertices(), GL_UNSIGNED_INT, tri->hostTriangleIndices());

	glDisableClientState(GL_VERTEX_ARRAY);
	
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	
	glColor3f(0.28f, 0.29f, 0.4f);
	
    glEnableClientState(GL_VERTEX_ARRAY);

	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)tri->hostX());
	glDrawElements(GL_TRIANGLES, tri->numTriangleFaceVertices(), GL_UNSIGNED_INT, tri->hostTriangleIndices());

	glDisableClientState(GL_VERTEX_ARRAY);
}

void AdeniumWorld::drawTetrahedron()
{
    if(!m_tetraMesh) {
        std::cout<<"\n WARNING: adenium world has tetrahedron mesh!\n";
        return;
    }
    
    glColor3f(0.13f, 0.29f, 0.24f);
    const unsigned nt = m_tetraMesh->numTetrahedrons();
    Vector3F * p = m_tetraDeformer->deformedP();
    glBegin(GL_TRIANGLES);
    unsigned i, j;
    Vector3F q;
    for(i=0; i< nt; i++) {
        unsigned * tet = m_tetraMesh->tetrahedronIndices(i);
        for(j=0; j< 12; j++) {
            q = p[ tet[ TetrahedronToTriangleVertex[j] ] ];
            glVertex3fv((GLfloat *)&q);
        }
    }
    glEnd();
}

void AdeniumWorld::drawAnchors()
{
    if(!m_tetraMesh) return;
    glEnable(GL_DEPTH_TEST);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glColor3f(.59f, .21f, 0.f);
    unsigned * a = m_tetraMesh->anchors();
    Vector3F * p = m_tetraDeformer->deformedP();
    unsigned i = 0;
    for(;i<m_tetraMesh->numPoints();i++) {
        if(a[i]>0) DbgDrawer->drawer()->alignedDisc(p[i], 0.08f);
    }
}

void AdeniumWorld::initOnDevice()
{
    CudaBase::SetGLDevice();
    CudaLinearBvh::Builder->initOnDevice();
	
	unsigned ne, np;
    unsigned i=0;
	for(;i < m_numObjects; i++) {
		CudaMassSystem * curObj = m_objects[i];
		
		ne = curObj->numElements();
		np = curObj->numPoints();
		
		m_objectPos[i] = new CUDABuffer;
		m_objectPos[i]->create(np*12);
		
		curObj->setDeviceXPtr(m_objectPos[i], 0);
		m_objectPos[i]->hostToDevice(curObj->hostX(), np * 12);
		
		m_objectVel[i] = new CUDABuffer;
		m_objectVel[i]->create(np*12);
        
        curObj->setDeviceVPtr(m_objectVel[i], 0);
		m_objectVel[i]->hostToDevice(curObj->hostV(), np * 12);
        
        m_objectAnchoredVel[i] = new CUDABuffer;
		m_objectAnchoredVel[i]->create(np*12);
		
		curObj->setDeviceVaPtr(m_objectVel[i], 0);
		m_objectAnchoredVel[i]->hostToDevice(curObj->hostV(), np * 12);
        
		m_objectInd[i] = new CUDABuffer;
		m_objectInd[i]->create(ne * 16);
		
		curObj->setDeviceTretradhedronIndicesPtr(m_objectInd[i], 0);
		m_objectInd[i]->hostToDevice(curObj->hostTetrahedronIndices(), ne * 16);
		
		curObj->initOnDevice();
        std::cout<<"\nobj update";
		m_objects[i]->update();
		m_objects[i]->sendDbgToHost();
	}
}

void AdeniumWorld::resizeRenderArea(int w, int h)
{
    if(!m_image->resize(w, h)) return;
	
	if(m_texture) glDeleteTextures(1, &m_texture);
	glGenTextures(1, &m_texture);
    glBindTexture(GL_TEXTURE_2D, m_texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, w, h, 0, GL_RGBA, GL_FLOAT, NULL);
}

void AdeniumWorld::render(BaseCamera * camera)
{
	if(m_numObjects<1) return;
    Matrix44F mt = camera->fSpace;
	mt.transpose();
	m_image->setModelViewMatrix(mt.v);
	m_image->reset();
	if(camera->isOrthographic()) {
		m_image->renderOrhographic(camera, m_objects[0]);
	}
	else {
		m_image->renderPerspective(camera, m_objects[0]);
	}
	
	if(!m_texture) {
		glGenTextures(1, &m_texture);
		glBindTexture(GL_TEXTURE_2D, m_texture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, m_image->imageWidth(), m_image->imageHeight(), 0, GL_RGBA, GL_FLOAT, NULL);
	}
	
	glEnable(GL_TEXTURE_2D);
	m_image->bindBuffer();
	glBindTexture(GL_TEXTURE_2D, m_texture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_image->imageWidth(), m_image->imageHeight(), GL_RGBA, GL_FLOAT,0);
	m_image->unbindBuffer();
}

void AdeniumWorld::setDifferenceObject(ATriangleMesh * m)
{ 
    m_deformedMesh = m;
    std::cout<<"init translation "<<m_difference->resetTranslation(m);
}

ATriangleMesh * AdeniumWorld::deformedMesh()
{ return m_deformedMesh; }

void AdeniumWorld::processFrame(bool toReset)
{
    if(toReset) m_difference->resetTranslation(m_deformedMesh);
    else m_difference->addTranslation(m_deformedMesh);
    
    const Vector3F t = m_difference->lastTranslation();
    Matrix44F mat; mat.setTranslation(t);
    mat.inverse();
    m_deformedMesh->moveIntoSpace(mat);
    
    m_difference->computeQ(m_deformedMesh);
    if(toReset) m_tetraDeformer->reset(m_deformedMesh);
    else m_tetraDeformer->solve(m_deformedMesh);
    
    saveVelocity(toReset);
    
    const unsigned nv = m_deformedMesh->numPoints();
    if(isRayCast()) {
        m_objectPos[0]->hostToDevice(m_deformedMesh->points(), nv * 12);
        m_objects[0]->update();
    }
    else {
        m_objects[0]->setHostX((float *)m_deformedMesh->points());
    }
}

void AdeniumWorld::drawOverallTranslation()
{
    if(m_difference->numTranslations() < 2) return;
    unsigned i = 1;
    for(;i<m_difference->numTranslations();i++) {
        DbgDrawer->drawer()->arrow(m_difference->getTranslation(i-1), 
            m_difference->getTranslation(i));
    }
}

bool AdeniumWorld::isRayCast() const
{ return m_enableRayCast; }

void AdeniumWorld::toggleRayCast()
{
	if(m_enableRayCast) m_enableRayCast = false;
	else m_enableRayCast = true;
}

const Vector3F AdeniumWorld::currentTranslation() const
{ return m_difference->lastTranslation(); }

bool AdeniumWorld::saveVelocityFramerange(AFrameRange * fr)
{ 
    if(!m_velocityOut->isOpened()) {
        std::cout<<"\n WARNING: adenium world cannot write velocity file frame range!\n";
        return false;
    }
    m_velocityOut->writeFrameRange(fr);
    return true;
}

void AdeniumWorld::beginRecordVelocity(AFrameRange * fr)
{
    if(!saveVelocityFramerange(fr)) return;
    m_velocityOut->beginCountNumFramesPlayed();
    m_velocityOut->createPoints(totalNumPoints());
}

void AdeniumWorld::saveVelocity(bool toReset)
{
    if(doneVelocityCaching()) return;
    
    recordT();
	recordP();
    if(toReset) {
		m_velocityOut->resetMaxSpeed();
		m_velocityOut->frameBegin();
	}
    
	m_velocityOut->writeFrameTranslationalVelocity();
    m_velocityOut->writeFrameVelocity();
	m_velocityOut->writeMaxSpeed();
    
    if(m_velocityOut->isFrameEnd()) m_velocityOut->frameBegin();
    else m_velocityOut->nextFrame();
    
    m_velocityOut->countNumFramesPlayed();
}

const unsigned AdeniumWorld::totalNumPoints() const
{
    unsigned n = m_deformedMesh->numPoints();
    if(m_tetraMesh) n += m_tetraMesh->numPoints();
    else std::cout<<"\n WARNING: adenium world has no tetrahedron mesh!\n";
    return n;
}

void AdeniumWorld::recordT()
{ m_velocityOut->setCurrentTranslation(m_difference->lastTranslation()); }

void AdeniumWorld::recordP()
{
    unsigned vdrift = 0;
    unsigned nv;
    if(m_tetraMesh) {
        nv = m_tetraMesh->numPoints();
        m_velocityOut->setCurrentP(m_tetraDeformer->deformedP(), nv, vdrift);
        vdrift = nv;
    }
    nv = m_deformedMesh->numPoints();
    m_velocityOut->setCurrentP(m_deformedMesh->points(), nv, vdrift);
}

bool AdeniumWorld::doneVelocityCaching() const
{ return m_velocityOut->allFramesPlayed(); }
//:~