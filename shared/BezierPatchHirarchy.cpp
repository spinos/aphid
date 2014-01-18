#include "BezierPatchHirarchy.h"
#include <BezierPatch.h>
#include <PointInsidePolygonTest.h>
#include <InverseBilinearInterpolate.h>
#include <BiLinearInterpolate.h>
int BezierPatchHirarchy::MaxBezierPatchHirarchyLevel = 3;
BezierPatchHirarchy::BezierPatchHirarchy()
{
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
	if(m_childIdx) delete[] m_childIdx;
	if(m_planes) delete[] m_planes;
	if(m_invbil) delete[] m_invbil;
	m_childIdx = 0;
	m_planes = 0;
	m_invbil = 0;
}

void BezierPatchHirarchy::create(BezierPatch * parent)
{
    int ne = 0;
    int npl = 1;
    for(int i = 0; i <= MaxBezierPatchHirarchyLevel; i++) {
        ne += npl;
        npl *= 4;
    }
	//std::cout<<" ne "<<ne;
    m_childIdx = new unsigned[ne];
	m_planes = new PointInsidePolygonTest[ne];
	m_invbil = new InverseBilinearInterpolate[ne];
    BezierPatch p = *parent;
    p.resetTexcoord();
    unsigned current = 0;
	unsigned start = 1;
    recursiveCreate(&p, 1, current, start);
}

void BezierPatchHirarchy::recursiveCreate(BezierPatch * parent, short level, unsigned current, unsigned & start)
{
	m_childIdx[current] = start;
	m_planes[current].setBBox(parent->controlBBox());
	m_planes[current].createEdges(parent->_contorlPoints[0], parent->_contorlPoints[3], parent->_contorlPoints[15], parent->_contorlPoints[12]);
	m_planes[current].setTexcoord(parent->_texcoords[0], parent->_texcoords[1], parent->_texcoords[3], parent->_texcoords[2]);
	m_invbil[current].setVertices(m_planes[current].vertex(0),m_planes[current].vertex(1), m_planes[current].vertex(3), m_planes[current].vertex(2));
	//std::cout<<"\n l/c/s "<<level<<" "<<current;
	//if(level <= MaxBezierPatchHirarchy) std::cout<<" "<<start<<"-"<<start+3;	

	if(level > MaxBezierPatchHirarchyLevel) return;
    BezierPatch children[4];
    parent->decasteljauSplit(children);
    parent->splitPatchUV(children);

	unsigned childStart = start;
    start += 4;
    recursiveCreate(&children[0], level+1, childStart, start);
	recursiveCreate(&children[1], level+1, childStart+1, start);
	recursiveCreate(&children[2], level+1, childStart+2, start);
	recursiveCreate(&children[3], level+1, childStart+3, start);
}

char BezierPatchHirarchy::isEmpty() const
{
	return m_childIdx == 0;
}

char BezierPatchHirarchy::endLevel(int level) const
{
    return level > MaxBezierPatchHirarchyLevel;
}

PointInsidePolygonTest * BezierPatchHirarchy::plane(unsigned idx) const
{
	return &m_planes[idx];
}

unsigned BezierPatchHirarchy::childStart(unsigned idx) const
{
    return m_childIdx[idx];
}

Vector2F BezierPatchHirarchy::restoreUV(unsigned idx, const Vector3F & p) const
{
	Vector2F uv = m_invbil[idx](p);
	BiLinearInterpolate bili;
	return bili.interpolate2(uv.x, uv.y, m_planes[idx]._texcoords);
}

void BezierPatchHirarchy::setRebuild()
{
	cleanup();
}
