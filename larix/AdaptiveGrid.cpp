#include "AdaptiveGrid.h"
#include <KdIntersection.h>
#include <GjkIntersection.h>
#include <BaseBuffer.h>
#include <bcc_common.h>

unsigned AdaptiveGrid::CellNeighbourInds::InvalidIndex = 1<<30;

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
// start at 8 cells per axis
    int level = 3;
    const int dim = 1<<level;
    int i, j, k;

    const float h = cellSizeAtLevel(level);
    const float hh = h * .5f;

    const Vector3F ori = origin() + Vector3F(hh, hh, hh) * .999f;
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
    while(needRefine && level <= maxLevel) {
        std::cout<<" n level "<<level<<" cell "<<numCells()<<"\n";
		refine(tree);
		level++;
		if(level < maxLevel) tagCellsToRefine(tree);
    }
	m_cellsToRefine->clear();
}

bool AdaptiveGrid::tagCellsToRefine(KdIntersection * tree)
{
    m_cellsToRefine->clear();
    
    sdb::CellHash * c = cells();
    BoundingBox box;

    c->begin();
    bool result = false;
    //unsigned count;
    unsigned i = 0;
    while(!c->end()) {
        box = cellBox(c->key(), c->value()->level);
        gjk::IntersectTest::SetA(box);
		//count = tree->countElementIntersectBox(box);
        
        if(tree->intersectBox(box)) {
            setCellToRefine(c->key(), c->value(), 1);
            result = true;
        }
        else 
            setCellToRefine(c->key(), c->value(), 0);
        
        i++;
	    c->next();   
	}
    
    if(!result) return result;
    
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
				subs = sample + Vector3F(hh * OctChildOffset[u][0], 
				hh * OctChildOffset[u][1], 
				hh * OctChildOffset[u][2]);
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

static const float Cell6NeighboursCenterOffset[6][3] = {
{ -1.f,   0.f,   0.f},
{  1.f,   0.f,   0.f},
{  0.f,  -1.f,   0.f},
{  0.f,   1.f,   0.f},
{  0.f,   0.f,  -1.f},
{  0.f,   0.f,   1.f}
};

static const float Cell24NeighboursCenterOffset[24][3] = {
{-.75f, -.25f, -.25f}, // x-axis
{ .75f, -.25f, -.25f}, 
{-.75f,  .25f, -.25f},
{ .75f,  .25f, -.25f},
{-.75f, -.25f,  .25f},
{ .75f, -.25f,  .25f}, 
{-.75f,  .25f,  .25f},
{ .75f,  .25f,  .25f},
{-.25f, -.75f, -.25f}, // y-axis
{-.25f,  .75f, -.25f},
{ .25f, -.75f, -.25f},
{ .25f,  .75f, -.25f},
{-.25f, -.75f,  .25f},
{-.25f,  .75f,  .25f},
{ .25f, -.75f,  .25f},
{ .25f,  .75f,  .25f},
{-.25f, -.25f, -.75f}, // z-axis
{-.25f, -.25f,  .75f},
{ .25f, -.25f, -.75f},
{ .25f, -.25f,  .75f},
{-.25f,  .25f, -.75f},
{-.25f,  .25f,  .75f},
{ .25f,  .25f, -.75f},
{ .25f,  .25f,  .75f}
};

bool AdaptiveGrid::check24NeighboursToRefine(unsigned k, const sdb::CellValue * v)
{ 
    const Vector3F sample = cellCenter(k);
    const float h = cellSizeAtLevel(v->level);
    int i = 0;
    for(;i<24;i++) {
        Vector3F q = sample + Vector3F(h * Cell24NeighboursCenterOffset[i][0],
                                       h * Cell24NeighboursCenterOffset[i][1],
                                       h * Cell24NeighboursCenterOffset[i][2]);
		if(isPInsideBound(q)) {
			unsigned code = mortonEncode(q);
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
        sample = parentCenter + Vector3F(hh * OctChildOffset[i][0], 
        hh * OctChildOffset[i][1], 
        hh * OctChildOffset[i][2]);
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
	const Vector3F center = cellCenter(code);
	Vector3F neighbourP;
	
	int i = 0;
	for(;i<6; i++) {
		neighbourP = neighbourCellCenter(i, center, cellSizeAtLevel(v->level));
		sdb::CellValue * cell = findCell(neighbourP);
		if(cell)
			dst->side(i)[0] = cell->index;
		else
			findFinerNeighbourCells(dst, i, center, cellSizeAtLevel(v->level));
	}
}

Vector3F AdaptiveGrid::neighbourCellCenter(int i, const Vector3F & p, float size) const
{ 
	return p + Vector3F(size * Cell6NeighboursCenterOffset[i][0],
						size * Cell6NeighboursCenterOffset[i][1],
						size * Cell6NeighboursCenterOffset[i][2]); 
}

void AdaptiveGrid::findFinerNeighbourCells(CellNeighbourInds * dst, int side,
								const Vector3F & center, float size)
{
	Vector3F neighbourP;
	int i = 0;
	for(;i<4;i++) {
		neighbourP = finerNeighbourCellCenter(i, side, center, size);
		sdb::CellValue * cell = findCell(neighbourP);
		if(cell) dst->side(side)[i] = cell->index;
	}
}

Vector3F AdaptiveGrid::finerNeighbourCellCenter(int i, int side, const Vector3F & p, float size) const
{
	const int idx = i + side * 4;
	return p + Vector3F(size * Cell24NeighboursCenterOffset[idx][0],
						size * Cell24NeighboursCenterOffset[idx][1],
						size * Cell24NeighboursCenterOffset[idx][2]);
}
//:~