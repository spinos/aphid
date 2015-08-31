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

unsigned BlockBccMeshBuilder::MinimumUGrid = 3;
unsigned BlockBccMeshBuilder::MinimumVGrid = 1;
unsigned BlockBccMeshBuilder::MinimumWGrid = 1;
unsigned BlockBccMeshBuilder::MaximumUGrid = 256;
unsigned BlockBccMeshBuilder::MaximumVGrid = 3;
unsigned BlockBccMeshBuilder::MaximumWGrid = 1;
	
BlockBccMeshBuilder::BlockBccMeshBuilder() 
{
    m_verticesPool = new CartesianGrid;
	m_boxes = NULL;
}
BlockBccMeshBuilder::~BlockBccMeshBuilder() 
{
    delete m_verticesPool;
	if(m_boxes) delete[] m_boxes;
}

void BlockBccMeshBuilder::build(GeometryArray * geos, 
					unsigned & ntet, unsigned & nvert, unsigned & nstripes)
{
	tetrahedronP.clear();
	tetrahedronInd.clear();
	pointDrifts.clear();
	indexDrifts.clear();
	
    const unsigned n = geos->numGeometries();
	if(m_boxes) delete[] m_boxes;
	m_boxes = new AOrientedBox[n];
	
    unsigned i=0;
    for(;i<n;i++) {
        pointDrifts.push_back(tetrahedronP.size());
        indexDrifts.push_back(tetrahedronInd.size());
		build((AOrientedBox *)geos->geometry(i));
		m_boxes[i] = *(AOrientedBox *)geos->geometry(i);
	}
	
	ntet = tetrahedronInd.size()/4;
	nvert = tetrahedronP.size();
	nstripes = n;
}

void BlockBccMeshBuilder::build(AOrientedBox * ob)
{
	int gx = ob->extent().x * 2.f / EstimatedGroupSize;
	if(gx > MaximumUGrid ) gx = MaximumUGrid;
	else if(gx < MinimumUGrid ) gx = MinimumUGrid;
	
	int gy = ob->extent().y * 2.f / EstimatedGroupSize;
	if(gy > MaximumVGrid ) gy = MaximumVGrid;
	else if(gy < MinimumVGrid ) gy = MinimumVGrid;
	
	int gz = ob->extent().z * 2.f / EstimatedGroupSize;
	if(gz > MaximumWGrid ) gz = MaximumWGrid;
	else if(gz < MinimumWGrid ) gz = MinimumWGrid;
	
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
	Vector3F endP[2];
	float lowerDist;
	unsigned anchorTri;
	Vector3F anchorPnt;
	float anchorX;
	BoundingBox ab;
	Matrix33F invspace;
	bool isLowerEnd;
	unsigned i;
	Geometry::ClosestToPointTestResult cls;
	for(i=0;i<n;i++) {
		AOrientedBox & box = m_boxes[i];
		invspace = box.orientation();
		invspace.inverse();
// find which end of the box x-axis is closer to grow mesh
		endP[0]  = box.majorPoint(true);
		cls.reset(endP[0], 1e8f);
		anchorMesh->closestToPoint(&cls);
		lowerDist = cls._distance;
		anchorTri = cls._icomponent;
		anchorPnt = endP[0] - box.majorVector(true) * EstimatedGroupSize * .5f;
		anchorX = -box.extent().x + EstimatedGroupSize * .9f;
		isLowerEnd = true;
		
		endP[1] = box.majorPoint(false);
		cls.reset(endP[1], 1e8f);
		anchorMesh->closestToPoint(&cls);
		
		if(cls._distance < lowerDist) {
			anchorTri = cls._icomponent;
			anchorPnt = endP[1] - box.majorVector(false) * EstimatedGroupSize * .5f;
			anchorX = box.extent().x - EstimatedGroupSize * .9f;
			isLowerEnd = false;
		}
		addAnchorByThreshold(mesh, i, invspace, box.center(), 
							anchorX, isLowerEnd, anchorTri);
	}
	// addAnchor(mesh, anchorMesh);
}

void BlockBccMeshBuilder::addAnchorByThreshold(ATetrahedronMesh * mesh, 
					unsigned istripe,
					const Matrix33F & invspace, 
					const Vector3F & center,
					float threshold,
					bool isLower,
					unsigned tri)
{
	unsigned tetMax = mesh->numTetrahedrons() * 4;
	if(istripe < indexDrifts.size() -1)
		tetMax = indexDrifts[istripe + 1];
		
	BoundingBox box;
	Vector3F q;
	unsigned i = indexDrifts[istripe];
	unsigned j;
	for(;i<tetMax;i+=4) {
		unsigned * tet = mesh->tetrahedronIndices(i/4);
		box.reset();
        for(j=0; j< 4; j++)
            box.expandBy(mesh->points()[tet[j]], 1e-3f); 
			
		q = invspace.transform(box.center() - center);
		if(isLower) {
			if(q.x > threshold) continue;
		}
		else {
			if(q.x < threshold) continue;
		}
		
		for(j=0; j< 4; j++)
			mesh->anchors()[tet[j]] = (1<<24 | tri);
	}
}
//:~