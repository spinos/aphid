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
#include <kd/VoxelEngine.h>
#include <sdb/VectorArray.h>
#include <kd/GridBuilder.h>

namespace aphid {

template<typename T, typename Tn>
class VoxelGrid : public CartesianGrid {

	sdb::VectorArray<Voxel> m_voxels;
	sdb::VectorArray<AOrientedBox> m_oboxes;
	int m_maxLevel;
	
public:
	struct BuildProfile {
		int _minNPrimsPerCell;
		int _maxLevel;
		bool _extractDOP;
		bool _xyDOP;
		bool _shellOnly;
		
		BuildProfile() {
			_minNPrimsPerCell = 3;
			_maxLevel = 6;
			_extractDOP = false;
			_xyDOP = false;
			_shellOnly = false;
		}
	};
	
	VoxelGrid();
	virtual ~VoxelGrid();
	
	void create(KdNTree<T, Tn > * tree, 
				const BoundingBox & b,
				BuildProfile * profile);
	
	int numVoxels() const;
	
	const sdb::VectorArray<Voxel> & voxels() const;
	sdb::VectorArray<Voxel> * voxelsR();
	
	const sdb::VectorArray<AOrientedBox> & dops() const;
	
protected:

private:
	void createVoxels(KdNTree<T, Tn > * tree, int level, BuildProfile * profile);
	
};

template<typename T, typename Tn>
VoxelGrid<T, Tn>::VoxelGrid() {}

template<typename T, typename Tn>
VoxelGrid<T, Tn>::~VoxelGrid() {}

template<typename T, typename Tn>
void VoxelGrid<T, Tn>::create(KdNTree<T, Tn > * tree, 
										const BoundingBox & b,
										BuildProfile * profile)
{
	std::cout<<"\n creating voxel grid\n max level "<<profile->_maxLevel<<"\n";
	m_voxels.clear();
	m_oboxes.clear();
	
	setBounding(b);
	
	GridBuilder<T, Tn> builder;
	builder.build(tree, b, profile->_maxLevel, profile->_minNPrimsPerCell);
	
	if(builder.numFinalLevelCells() < 1) return;
	
	sdb::CellHash * c = builder.finalLevelCells();
	c->begin();
	while(!c->end() ) {
		addCell(c->key(), c->value()->level, 1, 0);
		c->next();
	}
	
	std::cout<<"\n level"<<builder.finalLevel()<<" n cell "<<numCells();
	
	createVoxels(tree, builder.finalLevel(), profile );
}

template<typename T, typename Tn>
void VoxelGrid<T, Tn>::createVoxels(KdNTree<T, Tn > * tree, int level, BuildProfile * profile)
{
	std::cout<<"\n voxelize 0 %";
		
	int minPrims = 1<<20;
	int maxPrims = 0;
	int totalPrims = 0;
	int nPrims;
	int ncpc = numCells() / 100;
	if(ncpc<1) ncpc = 1;
	int ic = 0, ipc = 0;
	float hh;
	
	Vector3F sample;
	BoxIntersectContext box;
	VoxelEngine<cvx::Triangle, KdNode4 >::Profile vepf;
	vepf._tree = tree;
	// vepf._approxCell = true;
	if(profile->_xyDOP) vepf._orientAtXY = true;
	KdEngine eng;
	sdb::CellHash * c = cells();
    c->begin();
    while(!c->end()) {

		if(c->value()->level == level) {
			
		sample = cellCenter(c->key() );
		hh = cellSizeAtLevel(c->value()->level ) * 0.49995f;
		
		box.setMin(sample.x - hh, sample.y - hh, sample.z - hh);
		box.setMax(sample.x + hh, sample.y + hh, sample.z + hh);
		box.reset(profile->_minNPrimsPerCell, true);
        
		eng.intersectBox<T, Tn>(tree, &box);
		
		nPrims = box.numIntersect();
		bool toBuild = nPrims > profile->_minNPrimsPerCell - 1;
		if(profile->_shellOnly)
			toBuild = !cellHas6Neighbors(c->key(), c->value()->level );
			
		if(toBuild) {
		
		if(minPrims > nPrims ) minPrims = nPrims;
		if(maxPrims < nPrims ) maxPrims = nPrims;
		totalPrims += nPrims;
		
			VoxelEngine<cvx::Triangle, KdNode4 > veng;
			vepf._bbox = box;
				
			if(veng.build(&vepf) ) {
			
			if(profile->_extractDOP) m_oboxes.insert(veng.orientedBBox() );
		
		Voxel v;
		v.setPos(c->key(), c->value()->level );
			veng.extractColor(v);
			veng.extractContours(v);
			m_voxels.insert(v);
			}
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
	
	std::cout<<"\n voxel grid end\n n voxel "<<numVoxels()
		<<"\n n prims per cell min/max/average "<<minPrims
	<<" / "<<maxPrims<<" / "<<(float)totalPrims/(float)numVoxels();
	std::cout.flush();
}

template<typename T, typename Tn>
int VoxelGrid<T, Tn>::numVoxels() const
{ return m_voxels.size(); }

template<typename T, typename Tn>
const sdb::VectorArray<Voxel> & VoxelGrid<T, Tn>::voxels() const
{ return m_voxels; }

template<typename T, typename Tn>
sdb::VectorArray<Voxel> * VoxelGrid<T, Tn>::voxelsR()
{ return &m_voxels; }

template<typename T, typename Tn>
const sdb::VectorArray<AOrientedBox> & VoxelGrid<T, Tn>::dops() const
{ return m_oboxes; }

}
#endif        //  #ifndef VOXELGRID_H
