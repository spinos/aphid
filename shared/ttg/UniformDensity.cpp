/*
 *  UniformDensity.cpp
 *  
 *
 *  Created by jian zhang on 2/10/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#include "UniformDensity.h"
#include <math/Vector3F.h>
#include <math/BoundingBox.h>
#include <math/ATypes.h>

namespace aphid {

namespace ttg {

const int UniformDensity::EightSubCellCoord[8][3] = {
{-1,-1,-1},
{ 0,-1,-1},
{-1, 0,-1},
{ 0, 0,-1},
{-1,-1, 0},
{ 0,-1, 0},
{-1, 0, 0},
{ 0, 0, 0}
};

UniformDensity::UniformDensity() :
m_numFronts(0)
{ memset(m_dim, 0, 16); }

UniformDensity::~UniformDensity()
{}

const int& UniformDensity::numCells() const
{ return m_dim[3]; }

bool UniformDensity::isEmpty() const
{ return numCells() < 1; }
	
void UniformDensity::create(int M, int N, int P,
				const float* boxOrigin,
				const float& cellSize)
{	
	m_dim[0] = M + (M&1);
	m_dim[1] = N + (N&1);;
	m_dim[2] = P + (P&1);;
	m_dim[3] = m_dim[0] * m_dim[1] * m_dim[2];
	m_rho.reset(new Float4[m_dim[3]]);
	m_front.reset(new DensityFront[m_dim[3]]);
	setOriginAndCellSize(boxOrigin, cellSize);
	buildGraph();
}

void UniformDensity::buildGraph()
{
	sdb::Sequence<sdb::Coord2> edgeMap;
	
	for(int k=0;k<dimension()[2];++k) {
		for(int j=0;j<dimension()[1];++j) {
			for(int i=1;i<dimension()[0];++i) {
				
				sdb::Coord2 c = sdb::Coord2(cellInd(i-1,j,k),
										cellInd(i,j,k) ).ordered();
				if(!edgeMap.findKey(c) ) {
					edgeMap.insert(c);
				}
			}
		}
	}
	
	for(int k=0;k<dimension()[2];++k) {
		for(int i=0;i<dimension()[0];++i) {
			for(int j=1;j<dimension()[1];++j) {
				
				sdb::Coord2 c = sdb::Coord2(cellInd(i,j-1,k),
										cellInd(i,j,k) ).ordered();
				if(!edgeMap.findKey(c) ) {
					edgeMap.insert(c);
				}
			}
		}
	}
	
	for(int j=0;j<dimension()[1];++j) {
		for(int i=0;i<dimension()[0];++i) {
			for(int k=1;k<dimension()[2];++k) {
				
				sdb::Coord2 c = sdb::Coord2(cellInd(i,j,k-1),
										cellInd(i,j,k) ).ordered();
				if(!edgeMap.findKey(c) ) {
					edgeMap.insert(c);
				}
			}
		}
	}
	
	std::map<int, std::vector<int> > vvemap;
	
	int c = 0;
	edgeMap.begin();
	while(!edgeMap.end() ) {
	
		int v0 = edgeMap.key().x;
		vvemap[v0].push_back(c);
		
		int v1 = edgeMap.key().y;
		vvemap[v1].push_back(c);
		
		c++;
		edgeMap.next();
	}
	
	std::vector<int> edgeBegins;
	std::vector<int> edgeInds;
	
	int nvve = 0;
	std::map<int, std::vector<int> >::iterator it = vvemap.begin();
	for(;it!=vvemap.end();++it) {
		edgeBegins.push_back(nvve);
		
		pushIndices(it->second, edgeInds);
		nvve += (it->second).size();
		
		it->second.clear();
	}
    
    const int& nv = numCells();
	int ne = edgeMap.size();
	int ni = edgeInds.size();
	BaseDistanceField::create(nv, ne, ni);
	
	DistanceNode * dst = nodes();
	for(int k=0;k<dimension()[2];++k) {
		for(int j=0;j<dimension()[1];++j) {
			for(int i=0;i<dimension()[0];++i) {
				
				dst[cellInd(i,j,k)].pos = getCellCenter(i,j,k);
			}
		}
	}
	
	extractEdges(&edgeMap);
	extractEdgeBegins(edgeBegins);
	extractEdgeIndices(edgeInds);
    
    vvemap.clear();
	edgeBegins.clear();
	edgeInds.clear();
	
    setAllEdgeLength(cellSize() );
}

void UniformDensity::setOriginAndCellSize(const float* boxOrigin,
				const float& cellSize)
{
	m_originSize[0] = boxOrigin[0];
	m_originSize[1] = boxOrigin[1];
	m_originSize[2] = boxOrigin[2];
	m_originSize[3] = cellSize;
	m_originSize[4] = 1.f / cellSize;
	setZero();
}

void UniformDensity::setZero()
{
	memset(m_rho.get(), 0, numCells() * 16);
}

bool UniformDensity::accumulate(const float& val, const Vector3F& p)
{
	int i = (p.x - m_originSize[0]) * m_originSize[4];
	if(i<0 || i>= m_dim[0])
		return false;
		
	int j = (p.y - m_originSize[1]) * m_originSize[4];
	if(j<0 || j>= m_dim[1])
		return false;	
		
	int k = (p.z - m_originSize[2]) * m_originSize[4];
	if(k<0 || k>= m_dim[2])
		return false;
	
	Float4& d = m_rho[cellInd(i,j,k)];
	d.x += p.x;
	d.y += p.y;
	d.z += p.z;
	d.w += val;// std::cout<<" d ("<<i<<","<<j<<","<<k<<") "<<d.w;
		
	return true;
}

void UniformDensity::finish()
{
	const int& n = numCells();
	
	resetNodes(1e20f, sdf::StBackGround, sdf::StUnknown);
	
	for(int i=0;i<n;++i) {
		const float& ri = m_rho[i].w; 
		if(ri > 0.f) {
			m_rho[i].x /= ri;
			m_rho[i].y /= ri;
			m_rho[i].z /= ri;
			
			setNodeDistance(i, -.1f);
		}
	}
	
	int iFar = firstEmptyCellInd();
/// visit out nodes
	marchOutside(iFar);
/// unvisited nodes are inside
	setFarNodeInside();
	
	for(int i=0;i<n;++i) {
		if(m_rho[i].w < 1e-3 && nodes()[i].stat == sdf::StFar ) {
			m_rho[i].w = 1e6f;
		}
	}
	
	m_numFronts = 0;
	for(int k=1;k<m_dim[2];++k) {
		for(int j=1;j<m_dim[1];++j) {
			for(int i=1;i<m_dim[0];++i) {
				
				if(detectFront(&m_front[m_numFronts], i, j, k) ) {
					m_numFronts++;
				}
			}
		}
	}
	
}

int UniformDensity::firstEmptyCellInd() const
{
	const int& n = numCells();
	for(int i=0;i<n;++i) {
		if(m_rho[i].w < 1e-4f) {
			return i;
		}
	}
	std::cout<<"\n ERROR: UniformDensity has no empty cell.";
	return 0; 
}

bool UniformDensity::detectFront(DensityFront* dst, int i, int j, int k)
{
	int i0, j0, k0;
	for(int s=0;s<8;++s) {
		i0 = i + EightSubCellCoord[s][0];
		j0 = j + EightSubCellCoord[s][1];
		k0 = k + EightSubCellCoord[s][2];
		m_subDensity[s] = safeGetDensity(i0,j0,k0);
	}
	
	int countSolid = 0;
	for(int s=0;s<8;++s) {
		if(m_subDensity[s] > 1e-3f)
			countSolid++;
	}
	if(countSolid < 1 || countSolid > 7)
		return false;
		
	dst->_ind[0] = i;
	dst->_ind[1] = j;
	dst->_ind[2] = k;
	estimateNormal(dst);
	return true;
}

void UniformDensity::estimateNormal(DensityFront* dst) const
{
static const int dxSubCellCoord[4][2] = {
	{1,0},
	{3,2},
	{5,4},
	{7,6}
};

static const int dySubCellCoord[4][2] = {
	{2,0},
	{3,1},
	{6,4},
	{7,5}
};

static const int dzSubCellCoord[4][2] = {
	{4,0},
	{5,1},
	{6,2},
	{7,3}
};
	
	Vector3F dpdu(0.f, 0.f, 0.f);
	for(int i=0;i<4;++i) {
		dpdu.x += m_subDensity[dxSubCellCoord[i][1]] - m_subDensity[dxSubCellCoord[i][0]];
		dpdu.y += m_subDensity[dySubCellCoord[i][1]] - m_subDensity[dySubCellCoord[i][0]];
		dpdu.z += m_subDensity[dzSubCellCoord[i][1]] - m_subDensity[dzSubCellCoord[i][0]];
	}
	
	float nmll = dpdu.length();
	if(nmll < 1e-4f)
		dpdu.set(0.f,1.f,0.f);
	else
		dpdu /= nmll;
		
	memcpy(dst->_nml, &dpdu, 12);
}

const int* UniformDensity::dimension() const
{ return m_dim; }

const float& UniformDensity::cellSize() const
{ return m_originSize[3]; }

const float& UniformDensity::getDensity(int i, int j, int k) const
{ return m_rho[cellInd(i,j,k)].w; }

float UniformDensity::safeGetDensity(int i, int j, int k) const
{
	if(i<0 || i>=m_dim[0])
		return 0.f;
	if(j<0 || j>=m_dim[1])
		return 0.f;
	if(k<0 || k>=m_dim[2])
		return 0.f;
	float d = getDensity(i,j,k);
	//	std::cout<<" d "<<d;
		
	return d;
}

void UniformDensity::safeGetPositionDensity(float* dst, int i, int j, int k) const
{
	dst[3] =  safeGetDensity(i,j,k);
	if(dst[3] < 1e-3f || dst[3] > 1e5f) {
		memset(dst, 0, 16);
		return;
	}
	memcpy(dst, &m_rho[cellInd(i,j,k)], 12);
}

const int& UniformDensity::numFronts() const
{ return m_numFronts; }
	
void UniformDensity::getFront(BoundingBox& frontBx, Vector3F& frontNml,
						const int& i) const
{
	const DensityFront& fi = m_front[i];
	const int* ind = fi._ind;
	frontBx = getCellBox(ind[0], ind[1], ind[2]);
	const Vector3F strag(-0.5f * m_originSize[3], -0.5f * m_originSize[3], -0.5f * m_originSize[3]);
	frontBx.translate(strag);
	memcpy(&frontNml, fi._nml, 12);
}

Vector3F UniformDensity::getCellCenter(int i, int j, int k) const
{
	const float& h = m_originSize[3];
	return Vector3F(m_originSize[0] + h * i,
					m_originSize[1] + h * j,
					m_originSize[2] + h * k);
}

BoundingBox UniformDensity::getCellBox(int i, int j, int k) const
{ 
	const float& h = m_originSize[3];
	float x = m_originSize[0] + h * i;
	float y = m_originSize[1] + h * j;
	float z = m_originSize[2] + h * k;
	return BoundingBox(x, y, z, x + h, y + h, z + h); 
}

int UniformDensity::cellInd(int i, int j, int k) const
{ return k*m_dim[0]*m_dim[1] + j*m_dim[0] + i; }

void UniformDensity::aggregateFrontSamplePos(Vector3F& pos, const DensityFront& fi)
{
	float sumPos[4];
	memset(sumPos, 0, 16);
	int i0, j0, k0;
	for(int s=0;s<8;++s) {
		i0 = fi._ind[0] + EightSubCellCoord[s][0];
		j0 = fi._ind[1] + EightSubCellCoord[s][1];
		k0 = fi._ind[2] + EightSubCellCoord[s][2];
		float* posRho = &m_subDensity[s<<2];
		safeGetPositionDensity(posRho, i0,j0,k0);
		
		sumPos[0] += posRho[0] * posRho[3];
		sumPos[1] += posRho[1] * posRho[3];
		sumPos[2] += posRho[2] * posRho[3];
		sumPos[3] += posRho[3];
		
	}
	
	const float oneoverrho = 1.f / sumPos[3];
	pos.set(sumPos[0] * oneoverrho,
			sumPos[1] * oneoverrho,
			sumPos[2] * oneoverrho);
	
}

}

}
