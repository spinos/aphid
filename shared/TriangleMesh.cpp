#include "TriangleMesh.h"

TriangleMesh::TriangleMesh() {}

TriangleMesh::~TriangleMesh() {}

const TypedEntity::Type TriangleMesh::type() const
{ return TTriangleMesh; }

const unsigned TriangleMesh::numComponents() const
{ return numTriangles(); }

const unsigned TriangleMesh::numTriangles() const
{ return numIndices() / 3; }

const unsigned TriangleMesh::numTriangleFaceVertices() const
{ return numIndices(); }

const BoundingBox TriangleMesh::calculateBBox(unsigned icomponent) const
{
	Vector3F * p = points();
	unsigned * v = &indices()[icomponent*3];
	BoundingBox box;
	box.updateMin(p[v[0]]);
	box.updateMax(p[v[0]]);
	box.updateMin(p[v[1]]);
	box.updateMax(p[v[1]]);
	box.updateMin(p[v[2]]);
	box.updateMax(p[v[2]]);
	return box;
}