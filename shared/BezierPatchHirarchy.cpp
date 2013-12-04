#include "BezierPatchHirarchy.h"
#include <BezierPatch.h>
#define MaxBezierPatchHirarchy 4
BezierPatchHirarchy::BezierPatchHirarchy()
{
	m_elm = 0;
	m_childIdx = 0;
}

BezierPatchHirarchy::~BezierPatchHirarchy()
{
    cleanup();
}

void BezierPatchHirarchy::cleanup()
{
	if(m_elm) delete[] m_elm;
	if(m_childIdx) delete[] m_childIdx;
	m_elm = 0;
	m_childIdx = 0;
}

void BezierPatchHirarchy::create(BezierPatch * parent)
{
    int ne = 0;
    int npl = 1;
    for(int i = 0; i <=MaxBezierPatchHirarchy; i++) {
        ne += npl;
        npl *= 4;
    }
	//std::cout<<"\nne "<<ne;
    m_elm = new BezierPatch[ne];
    m_childIdx = new unsigned[ne];
    m_elm[0] = *parent;
    unsigned current = 0;
	unsigned start = 1;
    recursiveCreate(parent, 1, current, start);
}

void BezierPatchHirarchy::recursiveCreate(BezierPatch * parent, short level, unsigned & current, unsigned & start)
{
    if(level > MaxBezierPatchHirarchy) return;
    BezierPatch * children = m_elm;
    children += start;
    parent->decasteljauSplit(children);
    
    m_childIdx[current] = start;
	//std::cout<<"\n l/c/s "<<level<<" "<<current<<" "<<start<<"-"<<start+3;
    current++;
    start += 4;
    recursiveCreate(&children[0], level+1, current, start);
	recursiveCreate(&children[1], level+1, current, start);
	recursiveCreate(&children[2], level+1, current, start);
	recursiveCreate(&children[3], level+1, current, start);
}

char BezierPatchHirarchy::isEmpty() const
{
	return m_elm == 0;
}
