#include "Automobile.h"
#include "Mesh.h"
#include "ChassisMdl.h"
#include "FrontWheelMdl.h"
#include "BackWheelMdl.h"

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <OpenGL/glext.h>
#include <GLUT/glut.h>
#endif

#ifdef WIN32
#include <windows.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glext.h>
#endif

namespace caterpillar {
Automobile::Automobile() 
{
    m_chassisMesh = new Mesh;
    fillMesh(m_chassisMesh, 
        sChassisNumVertices, sChassisNumTriangleIndices,
        sChassisMeshTriangleIndices,
        sChassisMeshVertices, sChassisMeshNormals);
       
    m_wheelMesh[0] = new Mesh;
    fillMesh(m_wheelMesh[0], 
        sFrontWheelNumVertices, sFrontWheelNumTriangleIndices,
        sFrontWheelMeshTriangleIndices,
        sFrontWheelMeshVertices, sFrontWheelMeshNormals);
    
    m_wheelMesh[1] = new Mesh;
    fillMesh(m_wheelMesh[1], 
        sBackWheelNumVertices, sBackWheelNumTriangleIndices,
        sBackWheelMeshTriangleIndices,
        sBackWheelMeshVertices, sBackWheelMeshNormals);
}

Automobile::~Automobile() 
{
    delete m_chassisMesh;
    delete m_wheelMesh[0];
    delete m_wheelMesh[1];
}

void Automobile::fillMesh(Mesh * m, 
        const int & nv, const int & ntv, 
        const int * indices,
        const float * pos, const float * nor) const
{
    btVector3 * p = m->createVertexPos(nv);
    Vector3F * n = m->createVertexNormal(nv);
    int * idx = m->createTriangles(ntv / 3);
	
    int i;
    for(i = 0; i < nv; i++) {
        p[i] = btVector3(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
        n[i].set(nor[i * 3], nor[i * 3 + 1], nor[i * 3 + 2]);
    }
	for(i = 0; i < ntv; i++) {
        idx[i] = indices[i];
    }
}

void Automobile::render() 
{
    drawMesh(m_chassisMesh);
    drawMesh(m_wheelMesh[0]);
    drawMesh(m_wheelMesh[1]);
}

void Automobile::drawMesh(Mesh * m)
{
    glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(4, GL_FLOAT, 0, (GLfloat*)&m->vertexPos()[0][0]);
	
	glEnableClientState(GL_NORMAL_ARRAY);
	glColorPointer(3, GL_FLOAT, 0, (GLfloat*)m->vertexNormal());
	
	glDrawElements(GL_TRIANGLES, m->getNumTri() * 3, GL_UNSIGNED_INT, m->indices());

	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
}

}
