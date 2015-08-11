#include "CartesianGrid.h"
#include <iostream>
#include <Morton3D.h>
#include <BNode.h>
#include <BaseBuffer.h>

CartesianGrid::CartesianGrid()
{
    m_numCells = 0;
	sdb::TreeNode::MaxNumKeysPerNode = 512;
	sdb::TreeNode::MinNumKeysPerNode = 32;
	m_cellHash = new sdb::CellHash;
}

CartesianGrid::CartesianGrid(const BoundingBox & bound) 
{
    m_origin = bound.getMin();
    m_span = bound.getLongestDistance();
    m_gridH = m_span / 1024.0;
    m_numCells = 0;
	sdb::TreeNode::MaxNumKeysPerNode = 512;
	sdb::TreeNode::MinNumKeysPerNode = 32;
	m_cellHash = new sdb::CellHash;
}

CartesianGrid::~CartesianGrid() 
{
    delete m_cellHash;
}

void CartesianGrid::setBounding(float * originSpan)
{
    m_origin.set(originSpan[0], originSpan[1], originSpan[2]);
	m_span = originSpan[3];
	m_gridH = m_span / 1024.0;
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

const float CartesianGrid::span() const
{ return m_span; }

sdb::CellHash * CartesianGrid::cells()
{ return m_cellHash; }

const float CartesianGrid::cellSizeAtLevel(int level) const
{ return m_span / (float)(1<<level); }

const float CartesianGrid::gridSize() const
{ return m_gridH; }

const unsigned CartesianGrid::mortonEncode(const Vector3F & p) const
{
//	const Vector3F q = putIntoBound(p);
    const float h = gridSize();
// numerical inaccuracy
    unsigned x = (p.x - m_origin.x) /h;
    unsigned y = (p.y - m_origin.y) /h;
    unsigned z = (p.z - m_origin.z) /h;
	return encodeMorton3D(x, y, z);
}

void CartesianGrid::gridOfP(const Vector3F & p, unsigned & x,
									unsigned & y,
									unsigned & z) const
{
	const float h = gridSize();
    x = (p.x - m_origin.x) / h;
	if(x>1023) x= 1023;
    y = (p.y - m_origin.y) / h;
	if(y>1023) y= 1023;
    z = (p.z - m_origin.z) / h;
	if(z>1023) z= 1023;
}

void CartesianGrid::gridOfCell(unsigned & x,
									unsigned & y,
									unsigned & z,
									int level) const
{
	int d = 10 - level;
    int a = (1<<(d-1));
	
    x = x>>d;
    y = y>>d;
    z = z>>d;
    
    x = x<<d;
    y = y<<d;
    z = z<<d;
    
    x += a;
    y += a;
    z += a;
}

unsigned CartesianGrid::mortonEncodeLevel(const Vector3F & p, int level) const
{
	unsigned x, y, z;
	gridOfP(p, x, y ,z);
    gridOfCell(x, y, z, level);
	return encodeMorton3D(x, y, z);
}

sdb::CellValue * CartesianGrid::findGrid(unsigned code) const
{ return m_cellHash->find(code); }

sdb::CellValue * CartesianGrid::findCell(unsigned code) const
{
    if(code < 1) return 0;
    return m_cellHash->find(code); 
}

unsigned CartesianGrid::addGrid(const Vector3F & p)
{
    unsigned code = mortonEncode(p);
    
	sdb::CellValue * ind = new sdb::CellValue;
	ind->level = 10;
	ind->visited = 0;
	m_cellHash->insert(code, ind);
	
    m_numCells++;
	return code;
}

void CartesianGrid::addCell(unsigned code, int level, int visited, unsigned index)
{
	sdb::CellValue * ind = new sdb::CellValue;
	ind->level = level;
	ind->visited = visited;
	ind->index = index;
	m_cellHash->insert(code, ind);
	
    m_numCells++;
}

unsigned CartesianGrid::addCell(const Vector3F & p, int level)
{
    unsigned code = mortonEncodeLevel(p, level);
	sdb::CellValue * ind = new sdb::CellValue;
	ind->level = level;
	ind->visited = 0;
	m_cellHash->insert(code, ind);
	
    m_numCells++;
	return code;
}

void CartesianGrid::removeCell(unsigned code)
{
	m_cellHash->remove(code);
	
	m_numCells--;
}

// cell level < 10
const Vector3F CartesianGrid::cellCenter(unsigned code) const
{ return gridOrigin(code); }

BoundingBox CartesianGrid::cellBox(unsigned code, int level) const
{
	BoundingBox box;
	Vector3F l = cellCenter(code);
	float h = cellSizeAtLevel(level) * .5f;
	box.setMin(l.x - h, l.y - h, l.z - h);
	box.setMax(l.x + h, l.y + h, l.z + h);
	return box;
}

const Vector3F CartesianGrid::gridOrigin(unsigned code) const
{
    unsigned x, y, z;
    decodeMorton3D(code, x, y, z);
    return Vector3F(m_origin.x + m_gridH * x, 
					m_origin.y + m_gridH * y, 
					m_origin.z + m_gridH * z);
}

const Vector3F CartesianGrid::cellOrigin(unsigned code, int level) const
{
    float h = cellSizeAtLevel(level) * 0.5f;
    return cellCenter(code) - Vector3F(h, h, h);
}

void CartesianGrid::putPInsideBound(Vector3F & p) const
{
    if(p.x <= m_origin.x) p.x = m_origin.x + 1e-7;
    if(p.x >= m_origin.x + m_span) p.x = m_origin.x + m_span - 1e-7;
    if(p.y <= m_origin.y) p.y = m_origin.y + 1e-7;
    if(p.y >= m_origin.y + m_span) p.y = m_origin.y + m_span - 1e-7;
    if(p.z <= m_origin.z) p.z = m_origin.z + 1e-7;
    if(p.z >= m_origin.z + m_span) p.z = m_origin.z + m_span - 1e-7;
}

bool CartesianGrid::isPInsideBound(const Vector3F & p) const
{ return (p.x - m_origin.x < m_span
			&& p.y - m_origin.y < m_span
			&& p.z - m_origin.z < m_span); }
			
unsigned CartesianGrid::encodeNeighborCell(unsigned code, 
									int level,
									int dx, int dy, int dz)
{
	unsigned x, y, z;
    decodeMorton3D(code, x, y, z);
	int d = 10 - level;
	int a = (1<<d);
	int ax = x + dx * a;
	int ay = y + dy * a;
	int az = z + dz * a;
	if(ax < 0 || ax > 1023) return 0;
	if(ay < 0 || ay > 1023) return 0;
	if(az < 0 || az > 1023) return 0;
	return encodeMorton3D(ax, ay, az);
}
			
sdb::CellValue * CartesianGrid::findNeighborCell(unsigned code, 
									int level,
									int dx, int dy, int dz)
{ return findCell( encodeNeighborCell(code, level, dx, dy, dz) ); }

unsigned CartesianGrid::encodeFinerNeighborCell(unsigned code, 
									int level,
									int dx, int dy, int dz,
									int cx, int cy, int cz) const
{
	unsigned x, y, z;
    decodeMorton3D(code, x, y, z);
	int d = 10 - level;
	int a = (1<<d);
	int c = 1<<(10 - level - 2);
	int ax = x + dx * a + cx * c;
	int ay = y + dy * a + cy * c;
	int az = z + dz * a + cz * c;
	if(ax < 0 || ax > 1023) return 0;
	if(ay < 0 || ay > 1023) return 0;
	if(az < 0 || az > 1023) return 0;
	
	return encodeMorton3D(ax, ay, az);
}

sdb::CellValue * CartesianGrid::findFinerNeighborCell(unsigned code, 
									int level,
									int dx, int dy, int dz,
									int cx, int cy, int cz)
{ return findCell( encodeFinerNeighborCell(code, level, dx, dy, dz, cx, cy, cz) ); }

void CartesianGrid::printHash()
{
	m_cellHash->begin();
	while(!m_cellHash->end()) {
	    std::cout<<" "<<m_cellHash->key()<<":";
		std::cout<<" "<<m_cellHash->value()->level<<"\n";
	    m_cellHash->next();   
	}
}

void CartesianGrid::printGrids(BaseBuffer * dst)
{
    const unsigned n = numCells();
    dst->create(n*12);
    unsigned * xyz = (unsigned *)dst->data();
    sdb::CellHash * c = cells();
    c->begin();
    while(!c->end()) {
        decodeMorton3D(c->key(), xyz[0], xyz[1], xyz[2]);
        c->next();  
        xyz += 3;
    }
}
//:~