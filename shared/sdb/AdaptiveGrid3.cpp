#include "AdaptiveGrid3.h"

namespace aphid {

namespace sdb {
    
AdaptiveGridCell::AdaptiveGridCell() :
m_parentCell(NULL),
m_numNeighbors(0),
m_hasChild(false)
{}

AdaptiveGridCell::~AdaptiveGridCell()
{}

const bool & AdaptiveGridCell::hasChild() const
{ return m_hasChild; }

void AdaptiveGridCell::setHasChild()
{ m_hasChild = true; }

void AdaptiveGridCell::setParentCell(SelfPtrType x, const int & i)
{ m_parentCell = x; m_childI = i; }

void AdaptiveGridCell::clearNeighbors()
{ m_numNeighbors = 0; }

void AdaptiveGridCell::addNeighbor(SelfPtrType x)
{ m_neighbors[m_numNeighbors++] = x; }

const int & AdaptiveGridCell::numNeighbors() const
{ return m_numNeighbors; }

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
	m_divideAllChild = false;
}

void AdaptiveGridDivideProfle::setLevels(int minLevel, int maxLevel)
{
	m_minLevel = minLevel;
	m_maxLevel = maxLevel;
}

void AdaptiveGridDivideProfle::setToDivideAllChild(bool x)
{ m_divideAllChild = x; }

const int & AdaptiveGridDivideProfle::minLevel() const
{ return m_minLevel; }

const int & AdaptiveGridDivideProfle::maxLevel() const
{ return m_maxLevel; }

const bool & AdaptiveGridDivideProfle::toDivideAllChild() const
{ return m_divideAllChild; }

}

}
