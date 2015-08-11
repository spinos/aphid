#include "AdaptiveGrid.h"
#include <KdIntersection.h>
#include <GjkIntersection.h>
#include <BaseBuffer.h>
#include <Morton3D.h>

unsigned AdaptiveGrid::CellNeighbourInds::InvalidIndex = 1<<30;

AdaptiveGrid::AdaptiveGrid()
{
    m_cellsToRefine = new sdb::CellHash;
}

AdaptiveGrid::AdaptiveGrid(float * originSpan) :
	CartesianGrid(originSpan)
{
    m_cellsToRefine = new sdb::CellHash;
}

AdaptiveGrid::AdaptiveGrid(const BoundingBox & bound) :
    CartesianGrid(bound)
{
    m_cellsToRefine = new sdb::CellHash;
}

AdaptiveGrid::~AdaptiveGrid() 
{
    delete m_cellsToRefine;
}

void AdaptiveGrid::create(KdIntersection * tree, int maxLevel)
{
	m_maxLevel = maxLevel;
	std::cout<<"\n create adaptive grid max level "<<maxLevel;
// start at 8 cells per axis
    int level = 3;
    const int dim = 1<<level;
    int i, j, k;

    const float h = cellSizeAtLevel(level);
    const float hh = h * .5f;

    const Vector3F ori = origin() + Vector3F(hh, hh, hh);
    Vector3F sample, closestP;
    BoundingBox box;
    for(k=0; k < dim; k++) {
        for(j=0; j < dim; j++) {
            for(i=0; i < dim; i++) {
                sample = ori + Vector3F(h* (float)i, h* (float)j, h* (float)k);
                box.setMin(sample.x - hh, sample.y - hh, sample.z - hh);
                box.setMax(sample.x + hh, sample.y + hh, sample.z + hh);
                addCell(sample, level);
            }
        }
    }
    bool needRefine = tagCellsToRefine(tree);
    while(needRefine && level < maxLevel) {
        std::cout<<"\n level"<<level<<" n cell "<<numCells();
		refine(tree);
		level++;
		if(level < maxLevel) needRefine = tagCellsToRefine(tree);
    }
	m_cellsToRefine->clear();
    std::cout<<"\n level"<<level<<" n cell "<<numCells();
}

bool AdaptiveGrid::tagCellsToRefine(KdIntersection * tree)
{
    m_cellsToRefine->clear();
    
    sdb::CellHash * c = cells();
    BoundingBox box;

    c->begin();
    bool result = false;
    while(!c->end()) {
        box = cellBox(c->key(), c->value()->level);
        gjk::IntersectTest::SetA(box);
		
        if(tree->intersectBox(box)) {
            setCellToRefine(c->key(), c->value(), 1);
            result = true;
        }
        else 
            setCellToRefine(c->key(), c->value(), 0);
        
        c->next();   
	}
    
    if(!result) return result;
    
    tagCellsToRefineByNeighbours();
	tagCellsToRefineByNeighbours();
	tagCellsToRefineByNeighbours();
	
    return result;
}

void AdaptiveGrid::tagCellsToRefineByNeighbours()
{
	sdb::CellHash * c = cells();
	c->begin();
    while(!c->end()) {
        if(!cellNeedRefine(c->key())) {
            if(check24NeighboursToRefine(c->key(), c->value()))
                setCellToRefine(c->key(), c->value(), 1);
        }
        c->next();
    }
}

static const float Cell8ChildOffset[8][3] = {
{-1.f, -1.f, -1.f},
{-1.f, -1.f, 1.f},
{-1.f, 1.f, -1.f},
{-1.f, 1.f, 1.f},
{1.f, -1.f, -1.f},
{1.f, -1.f, 1.f},
{1.f, 1.f, -1.f},
{1.f, 1.f, 1.f}};

void AdaptiveGrid::refine(KdIntersection * tree)
{    
	int level1;
	float hh;
    int u;
    Vector3F sample, subs;
	unsigned k;

	m_cellsToRefine->begin();
	while (!m_cellsToRefine->end()) {
		sdb::CellValue * parentCell = m_cellsToRefine->value();
		if(parentCell->visited > 0) {
        
			k = m_cellsToRefine->key();
			
			level1 = parentCell->level + 1;
			hh = cellSizeAtLevel(level1) * .5f;
			sample = cellCenter(k);
			removeCell(k);
			for(u = 0; u < 8; u++) {
				subs = sample + Vector3F(hh * Cell8ChildOffset[u][0], 
				hh * Cell8ChildOffset[u][1], 
				hh * Cell8ChildOffset[u][2]);
				addCell(subs, level1);
			}
		}
		
		m_cellsToRefine->next();
    }
}

void AdaptiveGrid::setCellToRefine(unsigned k, const sdb::CellValue * v,
                                   int toRefine)
{
    sdb::CellValue * ind = new sdb::CellValue;
	ind->level = v->level;
	ind->visited = toRefine;
	m_cellsToRefine->insert(k, ind);
}

bool AdaptiveGrid::cellNeedRefine(unsigned k)
{ 
    sdb::CellValue * parentCell = m_cellsToRefine->find(k);
    if(!parentCell) {
        //std::cout<<"error: cannot find cell "<<k;
        return false;
    }
    return parentCell->visited > 0;
}

static const int Cell6NeighborOffsetI[6][3] = {
{-1, 0, 0},
{ 1, 0, 0},
{ 0,-1, 0},
{ 0, 1, 0},
{ 0, 0,-1},
{ 0, 0, 1},
};

static const int Cell24FinerNeighborOffsetI[24][3] = {
{ 1,-1,-1}, // left
{ 1, 1,-1},
{ 1,-1, 1},
{ 1, 1, 1},
{-1,-1,-1}, // right
{-1, 1,-1},
{-1,-1, 1},
{-1, 1, 1},
{-1, 1,-1}, // bottom
{ 1, 1,-1},
{-1, 1, 1},
{ 1, 1, 1},
{-1,-1,-1}, // top
{ 1,-1,-1},
{-1,-1, 1},
{ 1,-1, 1},
{-1,-1, 1}, // back
{ 1,-1, 1},
{-1, 1, 1},
{ 1, 1, 1},
{-1,-1,-1}, // front
{ 1,-1,-1},
{-1, 1,-1},
{ 1, 1,-1}
};

bool AdaptiveGrid::check24NeighboursToRefine(unsigned k, const sdb::CellValue * v)
{ 
    int i, j;
    for(i=0;i<6;i++) {
		for(j=0;j<4;j++) {
			unsigned code = encodeFinerNeighborCell(k,
												v->level,
									Cell6NeighborOffsetI[i][0], 
									Cell6NeighborOffsetI[i][1],
									Cell6NeighborOffsetI[i][2],
									Cell24FinerNeighborOffsetI[i * 4 + j][0],
									Cell24FinerNeighborOffsetI[i * 4 + j][1],
									Cell24FinerNeighborOffsetI[i * 4 + j][2]);
			if(cellNeedRefine(code)) return true;
		}
    }
    return false; 
}

bool AdaptiveGrid::multipleChildrenTouched(KdIntersection * tree,
                                 const Vector3F & parentCenter,
                                 float parentSize)
{
    const float hh = parentSize * .5f;
    BoundingBox box;
    Vector3F sample;
    int count = 0;
    int i = 0;
    for(; i < 8; i++) {
        sample = parentCenter + Vector3F(hh * Cell8ChildOffset[i][0], 
        hh * Cell8ChildOffset[i][1], 
        hh * Cell8ChildOffset[i][2]);
        box.setMin(sample.x - hh, sample.y - hh, sample.z - hh);
        box.setMax(sample.x + hh, sample.y + hh, sample.z + hh);
        gjk::IntersectTest::SetA(box);
        if(tree->intersectBox(box)) count++;
        if(count > 1) return true;
    }
    return false;
}

void AdaptiveGrid::findNeighbourCells(CellNeighbourInds * dst, unsigned code,
                                      sdb::CellValue * v)
{
    dst->reset();
	int side = 0;
	for(;side<6; side++) {
		sdb::CellValue * cell = findNeighborCell(code, 
									v->level,
									Cell6NeighborOffsetI[side][0], 
									Cell6NeighborOffsetI[side][1],
									Cell6NeighborOffsetI[side][2]);
		if(cell)
			dst->side(side)[0] = cell->index;
		else {
			cell = findNeighborCell(code, 
									v->level - 1,
									Cell6NeighborOffsetI[side][0], 
									Cell6NeighborOffsetI[side][1],
									Cell6NeighborOffsetI[side][2]);
			if(cell)
				dst->side(side)[0] = cell->index;
			else
				findFinerNeighbourCells(dst, side, code, v->level);
		}
	}
}

void AdaptiveGrid::findFinerNeighbourCells(CellNeighbourInds * dst, 
								int side,
								unsigned code,
								int level)
{
	int i = 0;
	for(;i<4;i++) {
		sdb::CellValue * cell = findFinerNeighborCell(code,
												level,
									Cell6NeighborOffsetI[side][0], 
									Cell6NeighborOffsetI[side][1],
									Cell6NeighborOffsetI[side][2],
									Cell24FinerNeighborOffsetI[side * 4 + i][0],
									Cell24FinerNeighborOffsetI[side * 4 + i][1],
									Cell24FinerNeighborOffsetI[side * 4 + i][2]);
		if(cell) dst->side(side)[i] = cell->index;
	}
}

sdb::CellValue * AdaptiveGrid::locateCell(const Vector3F & p) const
{
	int l = maxLevel();
	unsigned x, y, z;
	gridOfP(p, x, y ,z);
	gridOfCell(x, y, z, l);
	
	unsigned code = encodeMorton3D(x, y, z);
	sdb::CellValue * found = findCell(code);
	if(found) return found;
	
	while (l>3) {
		l--;
		gridOfCell(x, y, z, l);
	
		code = encodeMorton3D(x, y, z);
		found = findCell(code);
		if(found) return found;
	}
	
	return 0; 
}

int AdaptiveGrid::maxLevel() const
{ return m_maxLevel; }

void AdaptiveGrid::setMaxLevel(int x)
{ m_maxLevel = x; }
//:~