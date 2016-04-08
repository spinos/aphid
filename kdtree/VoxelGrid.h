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

#include <Quantization.h>
#include <VectorArray.h>
#include <GridBuilder.h>

namespace aphid {

struct Voxel {
/// color in rgba 32bit
	int m_color;
/// morton code of cell center in bound
	int m_pos;
/// level-ncontour-indexcontour packed into int
/// bit layout
/// 0-3 level			4bit 0-15
/// 4-5 n contour		2bit 0-3
/// 6-   offset to first contour
	int m_contour;
	
	void setColor(const float &r, const float &g,
					const float &b, const float &a)
	{ col32::encodeC(m_color, r, g, b, a); }
	
/// set pos first
	void setPos(const int & morton, const int & level)
	{ 
		m_pos = morton; 
		m_contour = level;
	}
	
	void setContour(const int & count, const int & offset)
	{ m_contour = m_contour | (offset<<6 | count<<4); }
	
	void getColor(float &r, float &g,
					float &b, float &a)
	{ col32::decodeC(r, g, b, a, m_color); }
	
	BoundingBox calculateBBox() const
	{
		unsigned x, y, z;
		decodeMorton3D(m_pos, x, y, z);
		float h = 1<< (9 - (m_contour & 15) );
		
		BoundingBox b((float)x-h, (float)y-h, (float)z-h,
				(float)x+h, (float)y+h, (float)z+h);
		b.expand(-.00003f);
		return b;
	}
	
	bool intersect(const Ray &ray, float *hitt0, float *hitt1) const
	{ return calculateBBox().intersect(ray, hitt0, hitt1); }
	
	Vector3F calculateNormal() const
	{ return Vector3F(0.f, 1.f, 0.f); }
	
	static std::string GetTypeStr()
	{ return "voxel"; }
	
};

struct Contour {
/// point-normal packed into int
/// bit layout
/// 0-4		gridx		5bit
/// 5-9		girdy		5bit
/// 10-14	girdz		5bit
/// 15 normal sign		1bit
/// 16-17 normal axis	2bit
/// 18-23 normal u		6bit
/// 24-29 normal v		6bit
	int m_data;
	
	void setNormal(const Vector3F & n)
	{ colnor30::encodeN(m_data, n); }
	
	void setPoint(const Vector3F & p,
					const Vector3F & o,
					const float & d)
	{
		Vector3F c((p.x - o.x)/d, (p.y - o.y)/d, (p.z - o.z)/d);
		colnor30::encodeC(m_data, c);
	}
	
};

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
