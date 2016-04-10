/*
 *  VoxelEngine.h
 *  testntree
 *
 *  Created by jian zhang on 4/9/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APHID_VOXEL_ENGINE_H
#define APHID_VOXEL_ENGINE_H
#include <CartesianGrid.h>
#include <PrincipalComponents.h>
#include <vector>

namespace aphid {

template<typename T, int NLevel = 3>
class VoxelEngine : public CartesianGrid {

/// container of primitives
	std::vector<T> m_prims;
	std::vector<Vector3F> m_fronts[6];
	PrincipalComponents<std::vector<Vector3F> > m_pcas[6];
	AOrientedBox m_cuts[6];
	AOrientedBox m_obox;
	
public:
	VoxelEngine();
	virtual ~VoxelEngine();
	
	void add(const T & x);
	void build();
	
/// access to primitives
	const std::vector<T> & prims() const;
	const std::vector<Vector3F> & fronts(int x) const;
	const AOrientedBox & cut(int x) const;
	const AOrientedBox & orientedBBox() const;
	
protected:
	int intersect(const BoundingBox * b);
	void findFrontFaces(int facing, const BoundingBox * b);
	bool findNearestHit(Vector3F & dst,
						const BoundingBox & b,
						const Vector3F & ro,
						const Vector3F & d) const;
	void calculateOBox();
						
private:

};

template<typename T, int NLevel>
VoxelEngine<T, NLevel>::VoxelEngine()
{}

template<typename T, int NLevel>
VoxelEngine<T, NLevel>::~VoxelEngine()
{}

template<typename T, int NLevel>
void VoxelEngine<T, NLevel>::add(const T & x)
{ m_prims.push_back(x); }

template<typename T, int NLevel>
void VoxelEngine<T, NLevel>::build()
{
	const float h = cellSizeAtLevel(NLevel);
    const float hh = h * .49995f;
	const int dim = 1<<NLevel;
	const Vector3F ori = origin() + Vector3F(hh, hh, hh);
    Vector3F sample;
	BoundingBox box;
	int i, j, k;
	for(k=0; k < dim; k++) {
        for(j=0; j < dim; j++) {
            for(i=0; i < dim; i++) {
                sample = ori + Vector3F(h* (float)i, h* (float)j, h* (float)k);
                box.setMin(sample.x - hh, sample.y - hh, sample.z - hh);
                box.setMax(sample.x + hh, sample.y + hh, sample.z + hh);
				
				if(intersect(&box) > 0 )
					addCell(sample, NLevel, 1);
            }
        }
    }
	
	const BoundingBox tight = calculateBBox();
	for(i=0; i < 6; i++) {
		findFrontFaces(i, &tight);
		if(m_fronts[i].size() > 3) 
			m_cuts[i] = m_pcas[i].analyze(m_fronts[i], m_fronts[i].size() );
		std::cout<<"\n front["<<i<<"] "<< m_fronts[i].size();
	}
	
	calculateOBox();
}

template<typename T, int NLevel>
int VoxelEngine<T, NLevel>::intersect(const BoundingBox * b)
{
	int nsect = 0;
	typename std::vector<T>::const_iterator it = m_prims.begin();
	for(;it!= m_prims.end(); ++it) {
		const BoundingBox & cb = (*it).calculateBBox();
		if(cb.intersect(*b) ) {
			bool isect = false;
			if(cb.inside(*b) ) {
				isect = true;
			}
			else {
				isect = it-> template exactIntersect<BoundingBox >(*b);
			}
			
			if(isect)
				nsect++;
		}
	}
	return nsect;
}

const float RightDirTable[6][3] = {
{ 0.f, 0.f, 1.f},
{ 0.f, 0.f,-1.f},
{-1.f, 0.f, 0.f},
{-1.f, 0.f, 0.f},
{ 1.f, 0.f, 0.f},
{-1.f, 0.f, 0.f}
};

const float UpDirTable[6][3] = {
{ 0.f, 1.f, 0.f},
{ 0.f, 1.f, 0.f},
{ 0.f, 0.f, 1.f},
{ 0.f, 0.f, -1.f},
{ 0.f, 1.f, 0.f},
{ 0.f, 1.f, 0.f}
};

const float DepthDirTable[6][3] = {
{ 1.f, 0.f, 0.f},
{-1.f, 0.f, 0.f},
{ 0.f, 1.f, 0.f},
{ 0.f,-1.f, 0.f},
{ 0.f, 0.f, 1.f},
{ 0.f, 0.f,-1.f}
};

template<typename T, int NLevel>
void VoxelEngine<T, NLevel>::findFrontFaces(int facing, const BoundingBox * b)
{
	const Vector3F bo = origin();
	const float & d = span();
	Vector3F right, up, depth, ori;
	right.set(RightDirTable[facing][0], RightDirTable[facing][1], RightDirTable[facing][2]);
	up.set(UpDirTable[facing][0], UpDirTable[facing][1], UpDirTable[facing][2]);
	depth.set(DepthDirTable[facing][0], DepthDirTable[facing][1], DepthDirTable[facing][2]);
		   
	if(facing == 0) {
		ori = bo;
	}
	else if(facing == 1) {
		ori.set(bo.x + d, bo.y, bo.z + d);
	}
	else if(facing == 2) {
		ori.set(bo.x + d, bo.y, bo.z);
	}
	else if(facing == 3) {
		ori.set(bo.x + d, bo.y + d, bo.z + d);
	}
	else if(facing == 4) {
		ori = bo;
	}
	else {
		ori.set(bo.x + d, bo.y, bo.z + d);
	}
	
	const float h = cellSizeAtLevel(NLevel);
	const float hh = h * .5f;
	ori += right * hh;
	ori += up * hh;
	ori += depth * hh;
	
	const int dim = 1<<NLevel;
	int i, j, k;
	for(k=0; k<dim; ++k) {
		for(j=0; j<dim; ++j) {
			Vector3F ro = ori + right * h * k + up * h * j;
			Vector3F rd;
			if(findNearestHit(rd, *b, ro, depth) ) {
				m_fronts[facing].push_back(rd - depth * hh);
			}
		}
	}
}

template<typename T, int NLevel>
bool VoxelEngine<T, NLevel>::findNearestHit(Vector3F & dst,
									const BoundingBox & b,
									const Vector3F & ro,
									const Vector3F & d) const
{
	if(findCell(ro, NLevel) ) return false;
	const float h = cellSizeAtLevel(NLevel);
	const int dim = 1<<NLevel;
	for(int i=1; i<dim; ++i) {
		dst = ro + d * i * h;
		if(!b.isPointInside(dst) ) continue;
		if(findCell(dst, NLevel) ) {
			return true;
		}
	}
	return false;		
}

template<typename T, int NLevel>
void VoxelEngine<T, NLevel>::calculateOBox()
{
	std::vector<Vector3F> pnts;
#if 0
	const float hh = cellSizeAtLevel(NLevel + 2);
	Vector3F cnt;
	sdb::CellHash * c = cells();
    c->begin();
    while(!c->end()) {
        
		cnt = cellCenter(c->key() );
		for(int i=0; i< 8; ++i)
			pnts.push_back(cnt + Vector3F(Cell8ChildOffset[i][0],
											Cell8ChildOffset[i][1],
											Cell8ChildOffset[i][2]) * hh );
			
        c->next();
	}
#else	
	BoundingBox b;
	getBounding(b);
	Vector3F p;
	
	typename std::vector<T>::const_iterator it = m_prims.begin();
	it = m_prims.begin();
	for(;it!= m_prims.end(); ++it) {
		p = (*it).P(0);
		if(b.isPointInside(p) )
			pnts.push_back(p);
			
		p = (*it).P(1);
		if(b.isPointInside(p ) )
			pnts.push_back(p);
			
		p = (*it).P(2);
		if(b.isPointInside(p ) )
			pnts.push_back(p);
	}
	
	for(int i=0; i< 100; ++i) {
		it = m_prims.begin();
		for(;it!= m_prims.end(); ++it) {
			const BoundingBox & cb = (*it).calculateBBox();
			if(cb.intersect(b) ) {
				if((*it).sampleP(p, b) )
					pnts.push_back(p);
			}
		}
	}
#endif
	std::cout<<"\n n pnt "<<pnts.size();
	PrincipalComponents<std::vector<Vector3F> > obpca;
	m_obox = obpca.analyze(pnts, pnts.size() );
}

template<typename T, int NLevel>
const std::vector<T> & VoxelEngine<T, NLevel>::prims() const
{ return m_prims; }

template<typename T, int NLevel>
const std::vector<Vector3F> & VoxelEngine<T, NLevel>::fronts(int x) const
{ return m_fronts[x]; }

template<typename T, int NLevel>
const AOrientedBox & VoxelEngine<T, NLevel>::cut(int x) const
{ return m_cuts[x]; }

template<typename T, int NLevel>
const AOrientedBox & VoxelEngine<T, NLevel>::orientedBBox() const
{ return m_obox; }

}

#endif
