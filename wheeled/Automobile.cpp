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

#include <DynamicsSolver.h>
#include <PhysicsState.h>

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
    Vector3F * p = m->createVertexPoint(nv);
    Vector3F * n = m->createVertexNormal(nv);
    int * idx = m->createTriangles(ntv / 3);
	
    int i;
    for(i = 0; i < nv; i++) {
        p[i].set(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
        n[i].set(nor[i * 3], nor[i * 3 + 1], nor[i * 3 + 2]);
    }
	for(i = 0; i < ntv; i++) {
        idx[i] = indices[i];
    }
}

void Automobile::render() 
{
    drawMesh(rigidBodyTM(getGroup("chassis")[0]), m_chassisMesh);
    drawMesh(rigidBodyTM(getGroup("wheel0")[0]), m_wheelMesh[0]);
    drawMesh(rigidBodyTM(getGroup("wheel0")[1]), m_wheelMesh[0]);
    drawMesh(rigidBodyTM(getGroup("wheel1")[0]), m_wheelMesh[1]);
    drawMesh(rigidBodyTM(getGroup("wheel1")[1]), m_wheelMesh[1]);
}

void Automobile::drawMesh(const Matrix44F & mat, Mesh * msh)
{
    float m[16];
	mat.glMatrix(m);
	
	glPushMatrix();
	glMultMatrixf((const GLfloat*)m);
	
    glColor3f(1.f, 1.f, 1.f);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)msh->vertexPoint());
	
	glEnableClientState(GL_NORMAL_ARRAY);
	glNormalPointer(GL_FLOAT, 0, (GLfloat*)msh->vertexNormal());
	
	glDrawElements(GL_TRIANGLES, msh->numTri() * 3, GL_UNSIGNED_INT, msh->indices());

	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	glPopMatrix();
}
}
