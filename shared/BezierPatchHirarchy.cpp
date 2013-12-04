#include "BezierPatchHirarchy.h"
#include <BezierPatch.h>
BezierPatchHirarchy::BezierPatchHirarchy() : m_elm(0)
{

}
BezierPatchHirarchy::~BezierPatchHirarchy()
{
    if(m_elm) delete[] m_elm;
}

void BezierPatchHirarchy::create(BezierPatch * parent, int maxLevel)
{
    m_maxLevel = maxLevel;
    int ne = 0;
    int npl = 1;
    for(int i = 0; i <=maxLevel; i++) {
        ne += npl;
        npl *= 4;
    }
    m_elm = new BezierPatch[ne];
    m_childIdx = new unsigned[ne];
    m_elm[0] = *parent;
    unsigned start = 1;
    unsigned current = 1;
    m_childIdx[0] = 1;
    recursiveCreate(parent, 0, current, start);
}

void BezierPatchHirarchy::recursiveCreate(BezierPatch * parent, int level, unsigned & current, unsigned & start)
{
    if(level > m_maxLevel) return;
    BezierPatch * children = m_elm;
    children += start;
    parent->decasteljauSplit(children);
    
    m_childIdx[current] = start;
    current++;
    start += 4;
    recursiveCreate(&children[0], level, current, start);
	recursiveCreate(&children[1], level, current, start);
	recursiveCreate(&children[2], level, current, start);
	recursiveCreate(&children[3], level, current, start);
}
