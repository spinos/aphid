#include "AdaptiveGrid3.h"

namespace aphid {

namespace sdb {
    
AdaptiveGridCell::AdaptiveGridCell() :
m_parentCell(NULL),
m_hasChild(false)
{
	clearNeighbors();
}

AdaptiveGridCell::~AdaptiveGridCell()
{}

const bool & AdaptiveGridCell::hasChild() const
{ return m_hasChild; }

void AdaptiveGridCell::setHasChild()
{ m_hasChild = true; }

void AdaptiveGridCell::setParentCell(SelfPtrType x, const int & i)
{ m_parentCell = x; m_childI = i; }

void AdaptiveGridCell::clearNeighbors()
{
	for(int i=0;i<26;++i) {
		m_neighbors[i] = 0;
	}
}

void AdaptiveGridCell::setNeighbor(SelfPtrType x, const int & i)
{ m_neighbors[i] = x; }

AdaptiveGridCell * AdaptiveGridCell::neighbor(const int & x)
{ return m_neighbors[x]; }

void AdaptiveGridCell::setChild(SelfPtrType x, const int & i)
{ m_children[i] = x; }

AdaptiveGridCell * AdaptiveGridCell::child(const int & i)
{ return m_children[i]; }

AdaptiveGridCell * AdaptiveGridCell::parentCell()
{ return m_parentCell; }

const int & AdaptiveGridCell::childInd() const
{ return m_childI; }

AdaptiveGridDivideProfle::AdaptiveGridDivideProfle()
{
	m_dividedCoord = 0;
	m_minLevel = 0;
	m_maxLevel = 5;
	m_minNormalDistribute = 1.f;
	m_divideAllChild = false;
	m_offset = .1f;
}

void AdaptiveGridDivideProfle::setLevels(int minLevel, int maxLevel)
{
	m_minLevel = minLevel;
	m_maxLevel = maxLevel;
}

void AdaptiveGridDivideProfle::setMinNormalDistribute(float x)
{ m_minNormalDistribute = x; }

void AdaptiveGridDivideProfle::setToDivideAllChild(bool x)
{ m_divideAllChild = x; }

void AdaptiveGridDivideProfle::setOffset(float x)
{ m_offset = x; }

const int & AdaptiveGridDivideProfle::minLevel() const
{ return m_minLevel; }

const int & AdaptiveGridDivideProfle::maxLevel() const
{ return m_maxLevel; }

const float & AdaptiveGridDivideProfle::minNormalDistribute() const
{ return m_minNormalDistribute; }

const bool & AdaptiveGridDivideProfle::toDivideAllChild() const
{ return m_divideAllChild; }

const float & AdaptiveGridDivideProfle::offset() const
{ return m_offset; }

}

}
