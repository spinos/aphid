#include "AdaptiveGrid.h"
#include <KdIntersection.h>
#include <GjkIntersection.h>
#include <BaseBuffer.h>
#include <bcc_common.h>

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
}

bool AdaptiveGrid::tagCellsToRefine(KdIntersection * tree)
{
    m_cellsToRefine->clear();
    
    sdb::CellHash * c = cells();
    Vector3F l;
    BoundingBox box;
    float h;
    c->begin();
    bool result = false;
    unsigned count;
    unsigned i = 0;
    while(!c->end()) {
        l = cellCenter(c->key());
		h = cellSizeAtLevel(c->value()->level) * .5f;
        box.setMin(l.x - h, l.y - h, l.z - h);
        box.setMax(l.x + h, l.y + h, l.z + h);
        
        gjk::IntersectTest::SetA(box);
		count = tree->countElementIntersectBox(box);
        
        if(count > 1) {
            setCellToRefine(c->key(), c->value(), 1);
            result = true;
        }
        else 
            setCellToRefine(c->key(), c->value(), 0);
        
        i++;
	    c->next();   
	}
    
    if(!result) return result;
    
    c->begin();
    while(!c->end()) {
        if(!cellNeedRefine(c->key())) {
            if(check24NeighboursToRefine(c->key(), c->value()))
                setCellToRefine(c->key(), c->value(), 1);
        }
        c->next();
    }
    return result;
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
{-.25f, -.25f, -.75f}, // x-axis
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
        unsigned code = mortonEncode(q);
        if(cellNeedRefine(code)) return true;
    }
    return false; 
}
