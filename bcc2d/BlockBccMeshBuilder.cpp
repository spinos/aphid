/*
 *  BlockBccMeshBuilder.cpp
 *  bcc
 *
 *  Created by jian zhang on 8/24/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "BlockBccMeshBuilder.h"
#include <CartesianGrid.h>
#include <tetrahedron_math.h>

BlockBccMeshBuilder::BlockBccMeshBuilder() 
{
    m_verticesPool = new CartesianGrid;
}
BlockBccMeshBuilder::~BlockBccMeshBuilder() 
{
    delete m_verticesPool;
}

void BlockBccMeshBuilder::build(GeometryArray * geos, 
					unsigned & ntet, unsigned & nvert, unsigned & nstripes)
{
	tetrahedronP.clear();
	tetrahedronInd.clear();
	pointDrifts.clear();
	indexDrifts.clear();
	
    const unsigned n = geos->numGeometries();
    unsigned i=0;
    for(;i<n;i++) {
        pointDrifts.push_back(tetrahedronP.size());
        indexDrifts.push_back(tetrahedronInd.size());
		build((AOrientedBox *)geos->geometry(i),
				8, 2, 1);
	}
	
	ntet = tetrahedronInd.size()/4;
	nvert = tetrahedronP.size();
	nstripes = n;
}

void BlockBccMeshBuilder::build(AOrientedBox * ob, 
				int gx, int gy, int gz)
{
    const Vector3F center = ob->center();
    const float span = ob->extent().x;
    float originSpan[4];
    originSpan[0] = center.x - span * 1.01f;
    originSpan[1] = center.y - span * 1.01f;
    originSpan[2] = center.z - span * 1.01f;
    originSpan[3] = span * 2.0202f;
    m_verticesPool->setBounding(originSpan);
    const float cellSize = span * 2.f / (float)gx;
    const float cellSizeH = cellSize * .5f;
    
    const Vector3F cellOrigin(center.x - cellSize * (float)gx * .5f + cellSizeH,
                        center.y - cellSize * (float)gy * .5f + cellSizeH,
                        center.z - cellSize * (float)gz * .5f + cellSizeH);
    Vector3F cellCenter;
    int i, j, k;
    for(k=0;k<gz;k++) {
        cellCenter.z = cellOrigin.z + cellSize * k;
        for(j=0;j<gy;j++) {
            cellCenter.y = cellOrigin.y + cellSize * j;
            for(i=0;i<gx;i++) {
                cellCenter.x = cellOrigin.x + cellSize * i;
				addNorth(cellCenter, cellSize, cellSizeH);
            }
        }
    }
	
	const Vector3F staggerOrigin = cellOrigin + Vector3F::XAxis * cellSizeH
									- Vector3F::YAxis * cellSizeH;
	
	for(k=0;k<gz;k++) {
        cellCenter.z = staggerOrigin.z + cellSize * k;
        for(j=0;j<gy+1;j++) {
            cellCenter.y = staggerOrigin.y + cellSize * j;
            for(i=0;i<gx-1;i++) {
                cellCenter.x = staggerOrigin.x + cellSize * i;
				addEast(cellCenter, cellSize, cellSizeH, j, gy);
            }
        }
    }
	
	if(gz>1) {
		const Vector3F fillOrigin = cellOrigin - Vector3F::YAxis * cellSizeH
									+ Vector3F::ZAxis * cellSizeH;
	
		for(k=0;k<gz-1;k++) {
			cellCenter.z = fillOrigin.z + cellSize * k;
			for(j=0;j<gy+1;j++) {
				cellCenter.y = fillOrigin.y + cellSize * j;
				for(i=0;i<gx;i++) {
					cellCenter.x = fillOrigin.x + cellSize * i;
					addDepth(cellCenter, cellSize, cellSizeH, j, gy);
				}
			}
		}
	}

// convert to index
	unsigned c = pointDrifts.back();
	sdb::CellHash * nodes = m_verticesPool->cells();
	nodes->begin();
	while(!nodes->end()) {
		nodes->value()->index = c;
		c++;
		nodes->next();
	}
	
	std::vector<unsigned >::iterator it = tetrahedronInd.begin();
	it += indexDrifts.back();
	for(;it!=tetrahedronInd.end();++it) {
		sdb::CellValue * found = m_verticesPool->findCell(*it);
		if(found) *it = found->index;
		else std::cout<<"\n error cannot find node "<<*it;
	}
	
// add p in world space
	Vector3F q;
	nodes->begin();
	while(!nodes->end()) {
		q = m_verticesPool->cellCenter(nodes->key());
		q -= ob->center();
		q = ob->orientation().transform(q);
		q += ob->center();
		tetrahedronP.push_back(q);
		nodes->next();
	}
}

void BlockBccMeshBuilder::addNorth(const Vector3F & cellCenter, float cellSize, float cellSizeH)
{
	Vector3F tetV[4];
	unsigned tetI[4];
	
	tetV[0] = cellCenter - Vector3F::YAxis * cellSizeH;
	tetV[1] = tetV[0] + Vector3F::YAxis * cellSize;
	tetV[2] = cellCenter - Vector3F::XAxis * cellSizeH
						- Vector3F::ZAxis * cellSizeH;
	tetV[3] = tetV[2] + Vector3F::ZAxis * cellSize;
	addTetrahedron(tetV, tetI);
	tetV[2] = tetV[3];
	tetV[3] = tetV[2] + Vector3F::XAxis * cellSize;
	addTetrahedron(tetV, tetI);
	tetV[2] = tetV[3];
	tetV[3] = tetV[2] - Vector3F::ZAxis * cellSize;
	addTetrahedron(tetV, tetI);
	tetV[2] = tetV[3];
	tetV[3] = tetV[2] - Vector3F::XAxis * cellSize;
	addTetrahedron(tetV, tetI);
}

void BlockBccMeshBuilder::addEast(const Vector3F & cellCenter, float cellSize, float cellSizeH, int i, int n)
{
	Vector3F tetV[4];
	unsigned tetI[4];
	
	tetV[0] = cellCenter - Vector3F::XAxis * cellSizeH;
	tetV[1] = tetV[0] + Vector3F::XAxis * cellSize;
	if(i==0) {
		tetV[2] = cellCenter + Vector3F::YAxis * cellSizeH
						- Vector3F::ZAxis * cellSizeH;
		tetV[3] = tetV[2] + Vector3F::ZAxis * cellSize;
		addTetrahedron(tetV, tetI);
	}
	else if(i==n) {
		tetV[2] = cellCenter - Vector3F::YAxis * cellSizeH
						+ Vector3F::ZAxis * cellSizeH;
		tetV[3] = tetV[2] - Vector3F::ZAxis * cellSize;
		addTetrahedron(tetV, tetI);
	}
	else {
		tetV[2] = cellCenter - Vector3F::YAxis * cellSizeH
						- Vector3F::ZAxis * cellSizeH;
		tetV[3] = tetV[2] + Vector3F::YAxis * cellSize;
		addTetrahedron(tetV, tetI);
		tetV[2] = tetV[3];
		tetV[3] = tetV[2] + Vector3F::ZAxis * cellSize;
		addTetrahedron(tetV, tetI);
		tetV[2] = tetV[3];
		tetV[3] = tetV[2] - Vector3F::YAxis * cellSize;
		addTetrahedron(tetV, tetI);
		tetV[2] = tetV[3];
		tetV[3] = tetV[2] - Vector3F::ZAxis * cellSize;
		addTetrahedron(tetV, tetI);
	}
}

void BlockBccMeshBuilder::addDepth(const Vector3F & cellCenter, float cellSize, float cellSizeH, int i, int n)
{
	Vector3F tetV[4];
	unsigned tetI[4];
	
	tetV[0] = cellCenter - Vector3F::ZAxis * cellSizeH;
	tetV[1] = tetV[0] + Vector3F::ZAxis * cellSize;
	if(i==0) {
		tetV[2] = cellCenter + Vector3F::XAxis * cellSizeH
							+ Vector3F::YAxis * cellSizeH;
		tetV[3] = tetV[2] - Vector3F::XAxis * cellSize;
		addTetrahedron(tetV, tetI);
	}
	else if(i==n) {
		tetV[2] = cellCenter - Vector3F::XAxis * cellSizeH
							- Vector3F::YAxis * cellSizeH;
		tetV[3] = tetV[2] + Vector3F::XAxis * cellSize;
		addTetrahedron(tetV, tetI);
	}
	else {
		tetV[2] = cellCenter - Vector3F::XAxis * cellSizeH
							- Vector3F::YAxis * cellSizeH;
		tetV[3] = tetV[2] + Vector3F::XAxis * cellSize;
		addTetrahedron(tetV, tetI);
		tetV[2] = tetV[3];
		tetV[3] = tetV[2] + Vector3F::YAxis * cellSize;
		addTetrahedron(tetV, tetI);
		tetV[2] = tetV[3];
		tetV[3] = tetV[2] - Vector3F::XAxis * cellSize;
		addTetrahedron(tetV, tetI);
		tetV[2] = tetV[3];
		tetV[3] = tetV[2] - Vector3F::YAxis * cellSize;
		addTetrahedron(tetV, tetI);
	}
}

void BlockBccMeshBuilder::addTetrahedron(Vector3F * v, unsigned * ind)
{
	int i = 0;
	for(;i<4;i++) ind[i] = m_verticesPool->addCell(v[i], 9);
// index as code for now
	tetrahedronInd.push_back(ind[0]);
	tetrahedronInd.push_back(ind[1]);
	tetrahedronInd.push_back(ind[2]);
	tetrahedronInd.push_back(ind[3]);
}

void BlockBccMeshBuilder::addAnchors(ATetrahedronMesh * mesh, unsigned n, KdIntersection * anchorMesh)
{

}
//:~