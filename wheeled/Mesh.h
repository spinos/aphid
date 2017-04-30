#pragma once
#include <Common.h>
namespace caterpillar {
class Mesh {
public:
    Mesh();
    virtual ~Mesh();

    btVector3 * createVertexPos(const int & nv);
    Vector3F * createVertexPoint(const int & nv);
    Vector3F * createVertexNormal(const int & nv);
	int * createTriangles(const int & ntri);
	
	const int numTri() const;
	const int numVert() const;
	
	int * indices();
	btVector3 * vertexPos();
	Vector3F * vertexPoint();
	Vector3F * vertexNormal();
private:
    btVector3 * m_vertexPos;
    Vector3F * m_vertexPoint;
    Vector3F * m_vertexNormal;
	int * m_indices;
	int m_numTri, m_numVert;
};
}
