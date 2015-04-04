#include "CartesianGrid.h"
#include <iostream>
#include <Morton3D.h>
CartesianGrid::CartesianGrid(const BoundingBox & bound, int maxLevel) 
{
    m_origin = bound.getMin();
    m_span = bound.getLongestDistance();
    
    const float margin = m_span / 41.f;
    m_origin.x -= margin;
    m_origin.y -= margin;
    m_origin.z -= margin;
    m_span += margin * 2.f;
    
// assuming 1.5625% cells are occupied
    m_maxNumCells = 1<<(maxLevel+maxLevel+maxLevel-6);
    
    m_cells = new CellIndex[m_maxNumCells];
    m_levels = new unsigned[m_maxNumCells];
    m_numCells = 0;
}

CartesianGrid::~CartesianGrid() 
{
    delete[] m_cells;
    delete[] m_levels;
}

const unsigned CartesianGrid::numCells() const
{ return m_numCells; }

void CartesianGrid::getBounding(BoundingBox & bound) const
{
    bound.setMin(m_origin.x, m_origin.y, m_origin.z);
    bound.setMax(m_origin.x + m_span, m_origin.y + m_span, m_origin.z + m_span);
}

const Vector3F CartesianGrid::origin() const
{ return m_origin; }

CartesianGrid::CellIndex * CartesianGrid::cells()
{ return m_cells; }

const float CartesianGrid::cellSizeAtLevel(int level) const
{ return m_span / (float)(1<<level); }

void CartesianGrid::setCell(unsigned i, const Vector3F & p, int level)
{
    const float h = m_span / 1024.f;
    int x = (p.x - m_origin.x) / h;
    int y = (p.y - m_origin.y) / h;
    int z = (p.z - m_origin.z) / h;
    unsigned code = encodeMorton3D(x, y, z);
    
    m_cells[i].key = code;
    m_cells[i].index = m_numCells;
    m_levels[i] = level;
}

void CartesianGrid::addCell(const Vector3F & p, int level)
{
    if(m_numCells >= m_maxNumCells) return;
    
    const float h = m_span / 1024.f;
    int x = (p.x - m_origin.x) / h;
    int y = (p.y - m_origin.y) / h;
    int z = (p.z - m_origin.z) / h;
    unsigned code = encodeMorton3D(x, y, z);
    
    m_cells[m_numCells].key = code;
    m_cells[m_numCells].index = m_numCells;
    m_levels[m_numCells] = level;
    m_numCells++;
}

const Vector3F CartesianGrid::cellCenter(unsigned i) const
{
    unsigned code = m_cells[i].key;
    unsigned x, y, z;
    decodeMorton3D(code, x, y, z);
    float h = m_span / 1024.f;
    return m_origin + Vector3F((x + .5f) * h, (y + .5f) * h, (z + .5f) * h);
}

