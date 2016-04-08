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
#include <vector>

namespace aphid {

struct PntNor {
	Vector3F m_p;
	Vector3F m_n;
};

template<typename T, int NLevel = 3>
class VoxelEngine : public CartesianGrid {

/// container of primitives
	std::vector<T> m_prims;
	std::vector<PntNor> m_fronts[6];
	
public:
	VoxelEngine();
	virtual ~VoxelEngine();
	
	void add(const T & x);
	void build();
	
/// access to primitives
	const std::vector<T> & prims() const;
	
protected:
	int intersect(const BoundingBox * b);
	void findFrontFaces(int facing, const BoundingBox * b);
	
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
	for(i=0; i < 6; i++)
		findFrontFaces(i, &tight);
	
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

template<typename T, int NLevel>
void VoxelEngine<T, NLevel>::findFrontFaces(int facing, const BoundingBox * b)
{
	
}

template<typename T, int NLevel>
const std::vector<T> & VoxelEngine<T, NLevel>::prims() const
{ return m_prims; }

}

#endif
