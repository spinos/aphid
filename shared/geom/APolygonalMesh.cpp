#include <geom/APolygonalMesh.h>
#include <math/BaseBuffer.h>
#include <geom/APolygonalUV.h>
#include <foundation/SHelper.h>

namespace aphid {

APolygonalMesh::APolygonalMesh() 
{
    m_faceCounts = new BaseBuffer;
    m_faceDrifts = new BaseBuffer;
}
APolygonalMesh::~APolygonalMesh() 
{
    delete m_faceCounts;
    delete m_faceDrifts;
	std::map<std::string, APolygonalUV * >::iterator it = m_uvs.begin();
	for(;it!=m_uvs.end();++it) {
		delete it->second;
	}
	m_uvs.clear();
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

void APolygonalMesh::addUV(const std::string & name, APolygonalUV * uv)
{ m_uvs[SHelper::getLastName(name)] = uv; }

const unsigned APolygonalMesh::numUVs() const
{ return m_uvs.size(); }

const std::string APolygonalMesh::uvName(unsigned idx) const
{ 
	unsigned i = 0;
	std::map<std::string, APolygonalUV * >::const_iterator it = m_uvs.begin();
	for(;it!=m_uvs.end();++it) {
		if(i==idx)
			return it->first;
		i++;
	}
	return "unknown";
}

APolygonalUV * APolygonalMesh::uvData(const std::string & name) const
{
	std::map<std::string, APolygonalUV * >::const_iterator it = m_uvs.begin();
	for(;it!=m_uvs.end();++it) {
		if(it->first==name)
			return it->second;
	}
	return 0; 
}

std::string APolygonalMesh::verbosestr() const
{
    std::stringstream sst;
    sst<<" polymesh nv "<<numPoints()
    <<"\n npoly "<<numPolygons()
    <<"\n nfacev "<<numIndices();
    if(numUVs() < 1) {
        sst<<"\n warning: mesh has no uv"
        <<"\n";
        return sst.str();
    }
    sst<<"\n nuvset "<<numUVs();
    unsigned i = 0;
    for(;i<numUVs();i++) {
        sst<<"\n uvset["<<i<<"] name "<<uvName(i);
        sst<<uvData(uvName(i))->verbosestr();
    }
    
    sst<<"\n";
    return sst.str();
}

}
//:~