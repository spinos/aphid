#include "Mesh.h"
namespace caterpillar {
Mesh::Mesh() 
{
	m_vertexPos = NULL;
	m_vertexPoint = NULL;
	m_vertexNormal = NULL;
	m_indices = NULL;
}

Mesh::~Mesh() 
{
	if(m_vertexPos) delete[] m_vertexPos;
	if(m_vertexPoint) delete[] m_vertexPoint;
	if(m_vertexNormal) delete[] m_vertexNormal;
	if(m_indices) delete[] m_indices;
}

btVector3 * Mesh::createVertexPos(const int & nv)
{
	if(m_vertexPos) delete[] m_vertexPos;
	m_vertexPos = new btVector3[nv];
	m_numVert = nv;
	return m_vertexPos;
}

Vector3F * Mesh::createVertexPoint(const int & nv)
{
	if(m_vertexPoint) delete[] m_vertexPoint;
	m_vertexPoint = new Vector3F[nv];
	m_numVert = nv;
	return m_vertexPoint;
}

Vector3F * Mesh::createVertexNormal(const int & nv)
{
    if(m_vertexNormal) delete[] m_vertexNormal;
	m_vertexNormal = new Vector3F[nv];
	m_numVert = nv;
	return m_vertexNormal;
}

int * Mesh::createTriangles(const int & ntri)
{
	if(m_indices) delete[] m_indices;
	m_numTri = ntri;
	m_indices = new int[ntri * 3];
	return m_indices;
}

const int Mesh::numTri() const { return m_numTri; }
const int Mesh::numVert() const { return m_numVert; }
int * Mesh::indices() { return m_indices; }
btVector3 * Mesh::vertexPos() { return m_vertexPos; }
Vector3F * Mesh::vertexPoint() { return m_vertexPoint; }
Vector3F * Mesh::vertexNormal() { return m_vertexNormal; }

}
