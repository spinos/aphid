/*
 *  UniformGrid.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 2/5/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "UniformGrid.h"
#include <KdIntersection.h>
#include <GjkIntersection.h>
#include <Morton3D.h>

UniformGrid::UniformGrid()
{
    m_cellsToRefine = new sdb::CellHash;
}

UniformGrid::~UniformGrid()
{
    delete m_cellsToRefine;
}

void UniformGrid::create(KdIntersection * tree, int maxLevel)
{
	m_maxLevel = maxLevel;
	std::cout<<"\n UniformGrid create max level "<<maxLevel;
// start at 8 cells per axis
    int level = 3;
    const int dim = 1<<level;
    int i, j, k;

    const float h = cellSizeAtLevel(level);
    const float hh = h * .5f;

    const Vector3F ori = origin() + Vector3F(hh, hh, hh);
    Vector3F sample;
    BoundingBox box;
    for(k=0; k < dim; k++) {
        for(j=0; j < dim; j++) {
            for(i=0; i < dim; i++) {
                sample = ori + Vector3F(h* (float)i, h* (float)j, h* (float)k);
                box.setMin(sample.x - hh, sample.y - hh, sample.z - hh);
                box.setMax(sample.x + hh, sample.y + hh, sample.z + hh);
				if(tree->intersectBox(box))
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

bool UniformGrid::tagCellsToRefine(KdIntersection * tree)
{
	m_cellsToRefine->clear();
	
    sdb::CellHash * c = cells();
    c->begin();
    while(!c->end()) {
        setCellToRefine(c->key(), c->value(), 1);
        c->next();
	}
    
    return true;
}

void UniformGrid::setCellToRefine(unsigned k, const sdb::CellValue * v,
                                   int toRefine)
{
    sdb::CellValue * ind = new sdb::CellValue;
	ind->level = v->level;
	ind->visited = toRefine;
	m_cellsToRefine->insert(k, ind);
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

void UniformGrid::refine(KdIntersection * tree)
{    
	int level1;
	float hh;
    Vector3F sample, subs;
	int u;
	unsigned k;
	BoundingBox box;
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
				box.setMin(subs.x - hh, subs.y - hh, subs.z - hh);
                box.setMax(subs.x + hh, subs.y + hh, subs.z + hh);
				if(tree->intersectBox(box)) 
					addCell(subs, level1);
			}
		}
		
		m_cellsToRefine->next();
    }
}

const int & UniformGrid::maxLevel() const
{ return m_maxLevel; }
