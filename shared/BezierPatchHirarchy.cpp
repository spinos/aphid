#include "BezierPatchHirarchy.h"
#include <BezierPatch.h>
#include <PointInsidePolygonTest.h>
#include <InverseBilinearInterpolate.h>
#include <BiLinearInterpolate.h>
#define MaxBezierPatchHirarchy 3
BezierPatchHirarchy::BezierPatchHirarchy()
{
	m_elm = 0;
	m_childIdx = 0;
	m_planes = 0;
	m_invbil = 0;
}

BezierPatchHirarchy::~BezierPatchHirarchy()
{
    cleanup();
}

void BezierPatchHirarchy::cleanup()
{
	if(m_elm) delete[] m_elm;
	if(m_childIdx) delete[] m_childIdx;
	if(m_planes) delete[] m_planes;
	if(m_invbil) delete[] m_invbil;
	m_elm = 0;
	m_childIdx = 0;
	m_planes = 0;
	m_invbil = 0;
}

void BezierPatchHirarchy::create(BezierPatch * parent)
{
    int ne = 0;
    int npl = 1;
    for(int i = 0; i <=MaxBezierPatchHirarchy; i++) {
        ne += npl;
        npl *= 4;
    }
    m_elm = new BezierPatch[ne];
    m_childIdx = new unsigned[ne];
	m_planes = new PointInsidePolygonTest[ne];
	m_invbil = new InverseBilinearInterpolate[ne];
    m_elm[0] = *parent;
    m_elm[0].resetTexcoord();
    unsigned current = 0;
	unsigned start = 1;
    recursiveCreate(1, current, start);
}

void BezierPatchHirarchy::recursiveCreate(short level, unsigned & current, unsigned & start)
{
	BezierPatch * parent = &m_elm[current];
	m_childIdx[current] = start;
	m_planes[current].createEdges(parent->_contorlPoints[0], parent->_contorlPoints[3], parent->_contorlPoints[15], parent->_contorlPoints[12]);
	m_invbil[current].setVertices(m_planes[current].vertex(0),m_planes[current].vertex(1), m_planes[current].vertex(3), m_planes[current].vertex(2));
	//std::cout<<"\n l/c/s "<<level<<" "<<current<<" "<<start<<"-"<<start+3;	
    current++;
	if(level > MaxBezierPatchHirarchy) return;
    BezierPatch * children = m_elm;
    children += start;
    parent->decasteljauSplit(children);
    parent->splitPatchUV(children);
    start += 4;
    recursiveCreate(level+1, current, start);
	recursiveCreate(level+1, current, start);
	recursiveCreate(level+1, current, start);
	recursiveCreate(level+1, current, start);
}

char BezierPatchHirarchy::isEmpty() const
{
	return m_elm == 0;
}

char BezierPatchHirarchy::endLevel(int level) const
{
    return level > MaxBezierPatchHirarchy;
}

BezierPatch * BezierPatchHirarchy::patch(unsigned idx) const
{
    return &m_elm[idx];
}

PointInsidePolygonTest * BezierPatchHirarchy::plane(unsigned idx) const
{
	return &m_planes[idx];
}

BezierPatch * BezierPatchHirarchy::childPatch(unsigned idx) const
{
    return &m_elm[m_childIdx[idx]];
}

unsigned BezierPatchHirarchy::childStart(unsigned idx) const
{
    return m_childIdx[idx];
}

Vector2F BezierPatchHirarchy::restoreUV(unsigned idx, const Vector3F & p) const
{
	Vector2F uv = m_invbil[idx](p);
	BiLinearInterpolate bili;
	return bili.interpolate2(uv.x, uv.y, m_elm[idx]._texcoords);
}
