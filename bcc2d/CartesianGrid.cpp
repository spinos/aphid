#include "CartesianGrid.h"
#include <iostream>
#include <Morton3D.h>
#include <BNode.h>
CartesianGrid::CartesianGrid(const BoundingBox & bound) 
{
    m_origin = bound.getMin();
    m_span = bound.getLongestDistance();
    
    const float margin = m_span / 43.f;
    m_origin.x -= margin;
    m_origin.y -= margin;
    m_origin.z -= margin;
    m_span += margin * 2.f;
	m_gridH = m_span / 1024.f;
    
    m_numCells = 0;
	
	sdb::TreeNode::MaxNumKeysPerNode = 256;
	sdb::TreeNode::MinNumKeysPerNode = 32;

	m_cellHash = new sdb::MortonHash;
}

CartesianGrid::~CartesianGrid() 
{
    delete m_cellHash;
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

sdb::MortonHash * CartesianGrid::cells()
{ return m_cellHash; }

const float CartesianGrid::cellSizeAtLevel(int level) const
{ return m_span / (float)(1<<level); }

const float CartesianGrid::gridSize() const
{ return m_gridH; }

void CartesianGrid::addCell(const Vector3F & p, int level)
{
    const float h = gridSize();
    int x = (p.x - m_origin.x) / h;
    int y = (p.y - m_origin.y) / h;
    int z = (p.z - m_origin.z) / h;
    unsigned code = encodeMorton3D(x, y, z);
    
	sdb::CellValue * ind = new sdb::CellValue;
	ind->level = level;
	m_cellHash->insert(code, ind);
	
    m_numCells++;
}

void CartesianGrid::removeCell(unsigned code)
{
	m_cellHash->remove(code);
	
	m_numCells--;
}

const Vector3F CartesianGrid::cellCenter(unsigned code) const
{
    float h = gridSize();
    return gridOrigin(code) + Vector3F(.5f * h, .5f * h, .5f * h);
}

const Vector3F CartesianGrid::gridOrigin(unsigned code) const
{
    unsigned x, y, z;
    decodeMorton3D(code, x, y, z);
    float h = gridSize();
    return m_origin + Vector3F(x * h, y * h, z * h);
}

const Vector3F CartesianGrid::cellOrigin(unsigned code, int level) const
{
    float h = cellSizeAtLevel(level) * 0.5f;
    return cellCenter(code) - Vector3F(h, h, h);
}

void CartesianGrid::printHash()
{
	m_cellHash->begin();
	while(!m_cellHash->end()) {
	    std::cout<<" "<<m_cellHash->key()<<":";
		std::cout<<" "<<m_cellHash->value()->level<<"\n";
	    m_cellHash->next();   
	}
}
