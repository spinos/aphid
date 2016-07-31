/*
 *  FieldTriangulation.cpp
 *  foo
 *
 *  Created by jian zhang on 7/25/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "FieldTriangulation.h"

using namespace aphid;

namespace ttg {

FieldTriangulation::FieldTriangulation()
{ 
	m_maxCutPosBuf = 0; 
	m_numFrontTris = 0;
	m_numFrontTriangleVertices = 0;
	m_cutPosBuf = NULL;
	m_triInds = NULL;
	m_vertexX = NULL;
	m_vertexN = NULL;
}

FieldTriangulation::~FieldTriangulation()
{
	if(m_maxCutPosBuf > 0) delete[] m_cutPosBuf;
	if(m_numFrontTris > 0) delete[] m_triInds;
	if(m_numFrontTriangleVertices > 0) {
		delete[] m_vertexX;
		delete[] m_vertexN; 
	}
}

const int & FieldTriangulation::numFrontTriangles() const
{ return m_numFrontTris; }

const Vector3F & FieldTriangulation::triangleP(const int & i, const int & j) const
{
	const sdb::Coord3 & c = m_triInds[i];
	
	int v;
	v = c.x;
	if(j==1)
		v = c.y;
	else if(j==2)
		v = c.z;
		
	if(v < MEncode)
		//return Vector3F::Zero;
		return nodes()[v].pos;
		
	return m_cutPosBuf[v & MDecode];
}

void FieldTriangulation::getTriangleShape(cvx::Triangle & t, const int & i) const
{	
	t.set(triangleP(i, 0),
			triangleP(i, 1),
			triangleP(i, 2));
}

void FieldTriangulation::getCutEdgeIndPos(int & cutInd,
				sdb::Array<sdb::Coord2, ICutEdge > & edgeMap,
				int & numCut,
				const RedBlueRefine & refiner,
				const int & iv0,
				const int & iv1,
				const DistanceNode * a,
				const DistanceNode * b)
{
	sdb::Coord2 k = sdb::Coord2(iv0, iv1).ordered();
	ICutEdge * e = edgeMap.find(k);
	if(e) {
		cutInd = e->ind;
	}
	else {
		e = new ICutEdge;
		e->key = k;
		cutInd = e->ind = numCut | MEncode;
		edgeMap.insert(k, e);
		
/// add a pnt
		m_cutPosBuf[numCut++] = refiner.splitPos(a->val, b->val, 
												a->pos, b->pos);
	}
}

void FieldTriangulation::cutEdges(RedBlueRefine & refiner,
					const ITetrahedron * t,
					const DistanceNode * tetn,
					sdb::Array<sdb::Coord2, ICutEdge > & edgeMap,
					int & numCut)
{
	if(!refiner.hasOption() )
		return;
	
	int icut;
	if(refiner.needSplitRedEdge(0) ) {
		getCutEdgeIndPos(icut, edgeMap, numCut,
							refiner, t->iv0, t->iv1, &tetn[0], &tetn[1]);
		refiner.splitRedEdge(0, icut, m_cutPosBuf[icut & MDecode]);
	}
	
	if(refiner.needSplitRedEdge(1) ) {
		getCutEdgeIndPos(icut, edgeMap, numCut,
							refiner, t->iv2, t->iv3, &tetn[2], &tetn[3]);
		refiner.splitRedEdge(1, icut, m_cutPosBuf[icut & MDecode]);
	}
	
	if(refiner.needSplitBlueEdge(0) ) {
		getCutEdgeIndPos(icut, edgeMap, numCut,
							refiner, t->iv0, t->iv2, &tetn[0], &tetn[2]);
		refiner.splitBlueEdge(0, icut, m_cutPosBuf[icut & MDecode]);
	}
	
	if(refiner.needSplitBlueEdge(1) ) {
		getCutEdgeIndPos(icut, edgeMap, numCut,
							refiner, t->iv0, t->iv3, &tetn[0], &tetn[3]);
		refiner.splitBlueEdge(1, icut, m_cutPosBuf[icut & MDecode]);
	}
	
	if(refiner.needSplitBlueEdge(2) ) {
		getCutEdgeIndPos(icut, edgeMap, numCut,
							refiner, t->iv1, t->iv2, &tetn[1], &tetn[2]);
		refiner.splitBlueEdge(2, icut, m_cutPosBuf[icut & MDecode]);
	}
	
	if(refiner.needSplitBlueEdge(3) ) {
		getCutEdgeIndPos(icut, edgeMap, numCut,
							refiner, t->iv1, t->iv3, &tetn[1], &tetn[3]);
		refiner.splitBlueEdge(3, icut, m_cutPosBuf[icut & MDecode]);
	}
}

void FieldTriangulation::triangulateFront()
{
	snapEdgeToFront();
	int nfe = countFrontEdges();

	if(m_maxCutPosBuf > 0) delete[] m_cutPosBuf;
/// three times as many as cut pnts
	m_maxCutPosBuf = nfe * 3;
	std::cout<<"\n n front edge "<<nfe<<" max n cut pos "<<m_maxCutPosBuf;
	
	m_cutPosBuf = new Vector3F[m_maxCutPosBuf];
	
	RedBlueRefine rbr;
	
	sdb::Array<sdb::Coord2, ICutEdge > edgeMap;
	sdb::Sequence<sdb::Coord3 > faceMap;
	DistanceNode tetn[4];
	int ncut = 0;
	 
	const int ntet = numTetrahedrons();
	int i=0;
	for(;i<ntet;++i) {
		const ITetrahedron * t = tetra(i);
		
		tetn[0] = nodes()[t->iv0];
		tetn[1] = nodes()[t->iv1];
		tetn[2] = nodes()[t->iv2];
		tetn[3] = nodes()[t->iv3];
		
		rbr.set(t->iv0, t->iv1, t->iv2, t->iv3);
		rbr.evaluateDistance(tetn[0].val, tetn[1].val, 
								tetn[2].val, tetn[3].val);
		rbr.estimateNormal(tetn[0].pos, tetn[1].pos, 
								tetn[2].pos, tetn[3].pos);
		
		cutEdges(rbr, t, tetn, edgeMap, ncut);
		rbr.refine();
		
		const int nft = rbr.numFrontTriangles();
		for(int j=0; j<nft; ++j) {
			const IFace * fj = rbr.frontTriangle(j);
			
			if(!faceMap.find(fj->key) ) {
				faceMap.insert(fj->key);
			}
		}
	}
	
	m_numAddedVert = ncut;
	std::cout<<" n cut "<<m_numAddedVert;
	
	dumpTriangleInd(faceMap);
	
	sdb::Array<int, int> vertMap;
	countTriangleVertices(vertMap);
	
	dumpVertex(vertMap);
	calculateVertexNormal(vertMap);
	dumpIndices(vertMap);
	
	edgeMap.clear();
	faceMap.clear();
	vertMap.clear();
	
	std::cout<<"\n n front triangle "<<m_numFrontTris
			<<"\n n front vertex "<<m_numFrontTriangleVertices;
}

void FieldTriangulation::dumpTriangleInd(sdb::Sequence<sdb::Coord3 > & faces)
{
	if(m_numFrontTris > 0) delete[] m_triInds; 
	m_numFrontTris = faces.size();
	m_triInds = new sdb::Coord3[m_numFrontTris];
	
	int i=0;
	faces.begin();
	while(!faces.end() ) {
	
		m_triInds[i++] = faces.key();

		faces.next();
	}
}

void FieldTriangulation::countTriangleVertices(sdb::Array<int, int> & vertMap)
{
	int i;
	for(i=0; i<m_numFrontTris; ++i) {
		const sdb::Coord3 & k = m_triInds[i];
		if(!vertMap.find(k.x))
			vertMap.insert(k.x, new int() );
		if(!vertMap.find(k.y))
			vertMap.insert(k.y, new int() );
		if(!vertMap.find(k.z))
			vertMap.insert(k.z, new int() );
	}
	
	i = 0;
	vertMap.begin();
	while(!vertMap.end() ) {
		*vertMap.value() = i++;
		vertMap.next();
	}
	
	if(m_numFrontTriangleVertices > 0) {
		delete[] m_vertexX;
		delete[] m_vertexN; 
	}
	m_numFrontTriangleVertices = i;
	m_vertexX = new Vector3F[i];
	m_vertexN = new Vector3F[i];
}

void FieldTriangulation::dumpVertex(aphid::sdb::Array<int, int> & vertMap)
{
	int i, j;
	for(i=0; i<m_numFrontTris; ++i) {
		const sdb::Coord3 & k = m_triInds[i];
		
		j = *vertMap.find(k.x);
		m_vertexX[j] = triangleP(i, 0);
		
		j = *vertMap.find(k.y);
		m_vertexX[j] = triangleP(i, 1);
		
		j = *vertMap.find(k.z);
		m_vertexX[j] = triangleP(i, 2);
	}
	
}

void FieldTriangulation::calculateVertexNormal(aphid::sdb::Array<int, int> & vertMap)
{
	cvx::Triangle atri;
	Vector3F triN;
	int i, j;
	for(i=0; i<m_numFrontTriangleVertices; ++i)
		m_vertexN[i].setZero();
		
	for(i=0; i<m_numFrontTris; ++i) {
		const sdb::Coord3 & k = m_triInds[i];
		
		getTriangleShape(atri, i);
		triN = atri.calculateNormal();
		//triN /= atri.calculateArea();
		
		j = *vertMap.find(k.x);
		m_vertexN[j] += triN;
		
		j = *vertMap.find(k.y);
		m_vertexN[j] += triN;
		
		j = *vertMap.find(k.z);
		m_vertexN[j] += triN;

	}
	
	for(i=0; i<m_numFrontTriangleVertices; ++i)
		m_vertexN[i].normalize();
}

void FieldTriangulation::dumpIndices(aphid::sdb::Array<int, int> & vertMap)
{
	int i, j;
	for(i=0; i<m_numFrontTris; ++i) {
		sdb::Coord3 & k = m_triInds[i];
		
		j = *vertMap.find(k.x);
		k.x = j;
		
		j = *vertMap.find(k.y);
		k.y = j;
		
		j = *vertMap.find(k.z);
		k.z = j;
		
	}
}

const int & FieldTriangulation::numAddedVertices() const
{ return m_numAddedVert; }

const Vector3F & FieldTriangulation::addedVertex(const int & i) const
{ return m_cutPosBuf[i]; }

const aphid::Vector3F * FieldTriangulation::triangleVertexP() const
{ return m_vertexX; }

const aphid::Vector3F * FieldTriangulation::triangleVertexN() const
{ return m_vertexN; }

const int & FieldTriangulation::numTriangleVertices() const
{ return m_numFrontTriangleVertices; }

const int * FieldTriangulation::triangleIndices() const
{ return (const int *)m_triInds; }

}