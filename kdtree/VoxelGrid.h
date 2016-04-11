/*
 *  VoxelGrid.h
 *  testntree
 *
 *  Created by jian zhang on 3/19/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 *  subdivide if intersected until max level
 */
#ifndef VOXELGRID_H
#define VOXELGRID_H
#include <VoxelEngine.h>
#include <VectorArray.h>
#include <GridBuilder.h>

namespace aphid {

template<typename T, typename Tn>
class VoxelGrid : public CartesianGrid {

	sdb::VectorArray<Voxel> m_voxels;
	sdb::VectorArray<Contour> m_contours;
	int m_maxLevel;
	
public:
	VoxelGrid();
	virtual ~VoxelGrid();
	
	void create(KdNTree<T, Tn > * tree, 
				const BoundingBox & b,
				int maxLevel = 8);
	
	int numVoxels() const;
	int numContours() const;
	
	const sdb::VectorArray<Voxel> & voxels() const;
	sdb::VectorArray<Voxel> * voxelsR();
	
protected:

private:
	void createVoxels(KdNTree<T, Tn > * tree, int level);
	
};

template<typename T, typename Tn>
VoxelGrid<T, Tn>::VoxelGrid() {}

template<typename T, typename Tn>
VoxelGrid<T, Tn>::~VoxelGrid() {}

template<typename T, typename Tn>
void VoxelGrid<T, Tn>::create(KdNTree<T, Tn > * tree, 
										const BoundingBox & b,
										int maxLevel)
{
	std::cout<<"\n creating voxel grid\n max level "<<maxLevel<<"\n";
	m_voxels.clear();
	m_contours.clear();
	
	setBounding(b);
	
	GridBuilder<T, Tn> builder;
	builder.build(tree, b, maxLevel);
	
	if(builder.numFinalLevelCells() < 1) return;
	
	sdb::CellHash * c = builder.finalLevelCells();
	c->begin();
	while(!c->end() ) {
		addCell(c->key(), c->value()->level, 1, 0);
		c->next();
	}
	
	std::cout<<"\n level"<<builder.finalLevel()<<" n cell "<<numCells();
	
	createVoxels(tree, builder.finalLevel() );
}

template<typename T, typename Tn>
void VoxelGrid<T, Tn>::createVoxels(KdNTree<T, Tn > * tree, int level)
{
	std::cout<<"\n voxelize 0 %";
		
	int minPrims = 1<<20;
	int maxPrims = 0;
	int totalPrims = 0;
	int nPrims;
	const int ncpc = numCells() / 100;
	int ic = 0, ipc = 0;
	float hh;
    Vector3F sample;
	BoxIntersectContext box;
	KdEngine eng;
	sdb::CellHash * c = cells();
    c->begin();
    while(!c->end()) {

		if(c->value()->level == level) {
			
		sample = cellCenter(c->key() );
		hh = cellSizeAtLevel(c->value()->level ) * 0.49995f;
		
		box.setMin(sample.x - hh, sample.y - hh, sample.z - hh);
		box.setMax(sample.x + hh, sample.y + hh, sample.z + hh);
		box.reset(1<<20, true);
        
		eng.intersectBox<T, Tn>(tree, &box);
		
		nPrims = box.numIntersect();
		if(nPrims > 0) {
		
		if(minPrims > nPrims ) minPrims = nPrims;
		if(maxPrims < nPrims ) maxPrims = nPrims;
		totalPrims += nPrims;
		
		Voxel v;
		v.setColor(.99f, .99f, .99f, .99f);
		v.setPos(c->key(), c->value()->level );
/// todo contours
		v.setContour(0, 0);
		m_voxels.insert(v);
		
		}
		}
		//else 
		//	 std::cout<<"\n waringing wrong level "<<c->value()->level<<" "<<c->key()<<std::endl;

		ic++;
		if(ic==ncpc) {
			ipc++;
			if((ipc & 3) == 3) {
				std::cout<<"\r voxelize "<<ipc<<" % ";
				std::cout.flush();
			}
			ic = 0;
		}
		
        c->next();
	}
	
	std::cout<<"\n n voxel "<<numVoxels()
		<<"\n n prims per cell min/max/average "<<minPrims
	<<" / "<<maxPrims<<" / "<<(float)totalPrims/(float)numVoxels();
}

template<typename T, typename Tn>
int VoxelGrid<T, Tn>::numVoxels() const
{ return m_voxels.size(); }

template<typename T, typename Tn>
int VoxelGrid<T, Tn>::numContours() const
{ return m_contours.size(); }

template<typename T, typename Tn>
const sdb::VectorArray<Voxel> & VoxelGrid<T, Tn>::voxels() const
{ return m_voxels; }

template<typename T, typename Tn>
sdb::VectorArray<Voxel> * VoxelGrid<T, Tn>::voxelsR()
{ return &m_voxels; }

}
#endif        //  #ifndef VOXELGRID_H
