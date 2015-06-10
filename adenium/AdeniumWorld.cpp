#include "AdeniumWorld.h"
#include <BvhTriangleSystem.h>
#include <GeoDrawer.h>
#include <CudaBase.h>
#include <BvhBuilder.h>
#include "AdeniumRender.h"
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

void AdeniumWorld::draw()
{
    unsigned i = 0;
    for(;i<m_numObjects; i++)
        drawTriangle(m_objects[i]);
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
    CudaBase::SetDevice();
    CudaLinearBvh::Builder->initOnDevice();
    unsigned i=0;
	for(;i < m_numObjects; i++) m_objects[i]->initOnDevice();
    m_image->initOnDevice();
}

void AdeniumWorld::resizeRenderArea(int w, int h)
{
    m_image->resize(w, h);
}
