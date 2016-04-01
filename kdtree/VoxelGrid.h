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

#include <CartesianGrid.h>
#include <IntersectionContext.h>
#include <Quantization.h>
#include <VectorArray.h>
#include <Morton3D.h>

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
		BoundingBox b;
		
		unsigned x, y, z;
		decodeMorton3D(m_pos, x, y, z);
		float h = 1<< (9 - (m_contour & 15) );
		h *= .9999f;
		
		return BoundingBox((float)x-h, (float)y-h, (float)z-h,
				(float)x+h, (float)y+h, (float)z+h);
				
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

template<typename Ttree, typename Tvalue>
class VoxelGrid : public CartesianGrid {

	sdb::VectorArray<Voxel> m_voxels;
	sdb::VectorArray<Contour> m_contours;
	int m_maxLevel;
	
public:
	VoxelGrid();
	virtual ~VoxelGrid();
	
	void create(Ttree * tree, 
				const BoundingBox & b,
				int maxLevel = 8);
	
	int numVoxels() const;
	int numContours() const;
	
	const sdb::VectorArray<Voxel> & voxels() const;
	sdb::VectorArray<Voxel> * voxelsR();
	
protected:

private:
	bool tagCellsToRefine(sdb::CellHash & cellsToRefine, int level);
	void refine(Ttree * tree, sdb::CellHash & cellsToRefine,
				int level);
	void createVoxels(Ttree * tree, int level);
	
};

template<typename Ttree, typename Tvalue>
VoxelGrid<Ttree, Tvalue>::VoxelGrid() {}

template<typename Ttree, typename Tvalue>
VoxelGrid<Ttree, Tvalue>::~VoxelGrid() {}

template<typename Ttree, typename Tvalue>
void VoxelGrid<Ttree, Tvalue>::create(Ttree * tree, 
										const BoundingBox & b,
										int maxLevel)
{
	std::cout<<"\n creating voxel grid\n max level "<<maxLevel<<"\n";
	m_voxels.clear();
	m_contours.clear();
	
	setBounding(b);
	m_maxLevel = maxLevel;
	
	int level = 3;
    const int dim = 1<<level;
    int i, j, k;

    const float h = cellSizeAtLevel(level);
    const float hh = h * .5f;
	
	const Vector3F ori = origin() + Vector3F(hh, hh, hh);
    Vector3F sample;
    BoxIntersectContext box;
    for(k=0; k < dim; k++) {
        for(j=0; j < dim; j++) {
            for(i=0; i < dim; i++) {
                sample = ori + Vector3F(h* (float)i, h* (float)j, h* (float)k);
                box.setMin(sample.x - hh, sample.y - hh, sample.z - hh);
                box.setMax(sample.x + hh, sample.y + hh, sample.z + hh);
				box.reset(1);
				tree->intersectBox(&box);
				if(box.numIntersect() > 0 )
					addCell(sample, level, 1);
            }
        }
    }
	
    while(level < m_maxLevel) {
		std::cout<<"\r level"<<level<<" n cell "<<numCells();
		std::cout.flush();
		sdb::CellHash * cellsToRefine = new sdb::CellHash;
		tagCellsToRefine(*cellsToRefine, level);
	
        refine(tree, *cellsToRefine, level);
		
		delete cellsToRefine;
		level++;
	}
	
	std::cout<<"\r level"<<level<<" n cell "<<numCells();
		
	createVoxels(tree, maxLevel);
}

template<typename Ttree, typename Tvalue>
bool VoxelGrid<Ttree, Tvalue>::tagCellsToRefine(sdb::CellHash & cellsToRefine,
												int level)
{
	sdb::CellHash * c = cells();
    c->begin();
    while(!c->end()) {
        
		if(c->value()->level == level) {
		sdb::CellValue * ind = new sdb::CellValue;
		ind->level = c->value()->level;
		ind->visited = 1;
		cellsToRefine.insert(c->key(), ind );
		
		}
        c->next();
	}
	return true;
}

template<typename Ttree, typename Tvalue>
void VoxelGrid<Ttree, Tvalue>::refine(Ttree * tree, sdb::CellHash & cellsToRefine,
										int level)
{	
	int level1;
	float hh;
    Vector3F sample, subs;
	int u;
	unsigned k;
	BoxIntersectContext box;
	
	cellsToRefine.begin();
	while (!cellsToRefine.end()) {
		k = cellsToRefine.key();
		sdb::CellValue * parentCell = cellsToRefine.value();
		
#if 0
		sdb::CellValue * old =  findCell(k);
		if(!old) std::cout<<"\n no old cell "<<k;
		else if(old->level != level) std::cout<<"\n waringing wrong old level "<<old->level<<" "<<level;
#endif

		level1 = parentCell->level + 1;
		hh = cellSizeAtLevel(level1) * .5;
		sample = cellCenter(k);
		
		for(u = 0; u < 8; u++) {
			subs = sample + Vector3F(hh * Cell8ChildOffset[u][0], 
			hh * Cell8ChildOffset[u][1], 
			hh * Cell8ChildOffset[u][2]);
			box.setMin(subs.x - hh, subs.y - hh, subs.z - hh);
			box.setMax(subs.x + hh, subs.y + hh, subs.z + hh);
			box.reset(1);
			tree->intersectBox(&box);
			if(box.numIntersect() > 0) 
				addCell(subs, level1, 1);
		}
		
		removeCell(k);
		
		cellsToRefine.next();
    }
}

template<typename Ttree, typename Tvalue>
void VoxelGrid<Ttree, Tvalue>::createVoxels(Ttree * tree, int level)
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
	sdb::CellHash * c = cells();
    c->begin();
    while(!c->end()) {
	
		if(c->value()->level == level) {
			
		sample = cellCenter(c->key() );
		hh = cellSizeAtLevel(c->value()->level ) * 0.5f;
		
		box.setMin(sample.x - hh, sample.y - hh, sample.z - hh);
		box.setMax(sample.x + hh, sample.y + hh, sample.z + hh);
		box.reset(1<<20, true);
        
		tree->intersectBox(&box);
		
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
		else 
			 std::cout<<"\n waringing wrong level "<<c->value()->level<<" "<<c->key()<<std::endl;

		ic++;
		if(ic==ncpc) {
			ipc++;
			if(!(ipc & 3)) {
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

template<typename Ttree, typename Tvalue>
int VoxelGrid<Ttree, Tvalue>::numVoxels() const
{ return m_voxels.size(); }

template<typename Ttree, typename Tvalue>
int VoxelGrid<Ttree, Tvalue>::numContours() const
{ return m_contours.size(); }

template<typename Ttree, typename Tvalue>
const sdb::VectorArray<Voxel> & VoxelGrid<Ttree, Tvalue>::voxels() const
{ return m_voxels; }

template<typename Ttree, typename Tvalue>
sdb::VectorArray<Voxel> * VoxelGrid<Ttree, Tvalue>::voxelsR()
{ return &m_voxels; }

}
#endif        //  #ifndef VOXELGRID_H
