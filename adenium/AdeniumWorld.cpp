#include "AdeniumWorld.h"
#include <BvhTriangleSystem.h>
#include <GeoDrawer.h>
#include <CudaBase.h>
#include <CUDABuffer.h>
#include <BvhBuilder.h>
#include <PerspectiveCamera.h>
#include "AdeniumRender.h"
#include <WorldDbgDraw.h>

WorldDbgDraw * AdeniumWorld::DbgDrawer = 0;
GLuint AdeniumWorld::m_texture = 0;

AdeniumWorld::AdeniumWorld() :
m_numObjects(0)
{
    m_image = new AdeniumRender;
}

AdeniumWorld::~AdeniumWorld() 
{
    delete m_image;
}

void AdeniumWorld::addTriangleSystem(BvhTriangleSystem * tri)
{
    if(m_numObjects == 32) return;
    m_objects[m_numObjects] = tri;
    m_numObjects++;
}

void AdeniumWorld::setBvhBuilder(BvhBuilder * builder)
{ CudaLinearBvh::Builder = builder; }

void AdeniumWorld::draw(BaseCamera * camera)
{
    unsigned i = 0;
    for(;i<m_numObjects; i++)
        drawTriangle(m_objects[i]);
		
	dbgDraw();
}

void AdeniumWorld::dbgDraw()
{
	if(!DbgDrawer) return;
    unsigned i;
#if DRAW_BVH_HIERARCHY
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
		
		m_objectInd[i] = new CUDABuffer;
		m_objectInd[i]->create(ne * 16);
		
		curObj->setDeviceTretradhedronIndicesPtr(m_objectInd[i], 0);
		m_objectInd[i]->hostToDevice(curObj->hostTetrahedronIndices(), ne * 16);
		
		curObj->initOnDevice();
		
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
		//m_image->renderPerspective(camera);
	}
	// m_image->sendToHost();
	
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
//:~