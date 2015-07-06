#include "APolygonalMesh.h"
#include "BaseBuffer.h"
APolygonalMesh::APolygonalMesh() 
{
    m_faceCounts = new BaseBuffer;
    m_faceDrifts = new BaseBuffer;
}
APolygonalMesh::~APolygonalMesh() 
{
    delete m_faceCounts;
    delete m_faceDrifts;
}

const TypedEntity::Type APolygonalMesh::type() const
{ return TPolygonMesh; }

const unsigned APolygonalMesh::numComponents() const
{ return numPolygons(); }

const unsigned APolygonalMesh::numPolygons() const
{ return m_numPolygons; }

unsigned * APolygonalMesh::faceCounts() const
{ return (unsigned *)m_faceCounts->data(); }

unsigned * APolygonalMesh::faceDrifts() const
{ return (unsigned *)m_faceDrifts->data(); }

unsigned * APolygonalMesh::polygonIndices(unsigned idx) const
{ return &indices()[faceDrifts()[idx]]; }

unsigned APolygonalMesh::faceCount(unsigned idx) const
{ return faceCounts()[idx]; }

void APolygonalMesh::create(unsigned np, unsigned ni, unsigned nf)
{
    createBuffer(np, ni);
	setNumPoints(np);
	setNumIndices(ni);
    
    m_numPolygons = nf;
    m_faceCounts->create(nf * 4);
    m_faceDrifts->create(nf * 4);
}

const BoundingBox APolygonalMesh::calculateBBox(unsigned icomponent) const
{
    Vector3F * p = points();
	unsigned * v = polygonIndices(icomponent);
    const unsigned n = faceCount(icomponent);
	BoundingBox box;
    unsigned i = 0;
    for(;i<n; i++) {
        box.updateMin(p[v[i]]);
        box.updateMax(p[v[i]]);
	}
	return box;
}

void APolygonalMesh::computeFaceDrift()
{
    unsigned * dst = faceDrifts();
    unsigned * src = faceCounts();
    dst[0] = 0;
    unsigned i = 1;
    for(;i<numPolygons();i++)
        dst[i] = dst[i-1] + src[i-1];
}
//:~
