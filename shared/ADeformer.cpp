#include "ADeformer.h"
#include "BaseBuffer.h"
#include "AGenericMesh.h"

ADeformer::ADeformer()
{
	m_deformedP = new BaseBuffer;
}

ADeformer::~ADeformer() 
{ delete m_deformedP; }

Vector3F * ADeformer::deformedP() const
{ return (Vector3F *)m_deformedP->data(); }

Vector3F * ADeformer::restP() const
{ return m_mesh->points(); }

void ADeformer::setMesh(AGenericMesh * mesh)
{
	m_mesh = mesh;
	const unsigned n = mesh->numPoints();
	m_deformedP->create(n*12);
	reset();
}

void ADeformer::reset()
{
	const unsigned n = numVertices();
	m_deformedP->copyFrom(m_mesh->points(), n*12);
}

unsigned ADeformer::numVertices() const
{ return m_mesh->numPoints(); }

const BoundingBox ADeformer::calculateBBox() const
{
	const unsigned n = numVertices();
	Vector3F * p = deformedP();
    BoundingBox b;
    for(unsigned i = 0; i < n; i++) {
        b.updateMin(p[i]);
		b.updateMax(p[i]);
	}
    return b;
}

bool ADeformer::solve() { return true; }

AGenericMesh * ADeformer::mesh()
{ return m_mesh; }
//:~