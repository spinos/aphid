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


}

}
