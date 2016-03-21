/*
 *  VoxelGrid.h
 *  testntree
 *
 *  Created by jian zhang on 3/19/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 *  subdivide if intersected until max level
 */

#include <CartesianGrid.h>
#include <IntersectionContext.h>
#include <Quantization.h>
#include <VectorArray.h>

namespace aphid {

struct Voxel {
/// color in rgba 32bit
	int m_color;
/// morton code in bound
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
	
	void setPos(const int & morton, const int & level)
	{ 
		m_pos = morton; 
		m_contour = m_contour | level;
	}
	
	void setContour(const int & count, const int & offset)
	{ m_contour = m_contour | (count<<4 | offset<<6); }
	
	void getColor(float &r, float &g,
					float &b, float &a)
	{ col32::decodeC(r, g, b, a, m_color); }
	
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
				int maxLevel = 9,
				int minLevel = 5);
	
	int numVoxels() const;
	int numContours() const;
	
protected:

private:
	bool tagCellsToRefine(sdb::CellHash & cellsToRefine);
	bool tagLevelCellsToRefine(sdb::CellHash & cellsToRefine, const int & level);
	void refine(Ttree * tree, sdb::CellHash & cellsToRefine);
	void createVoxels(Ttree * tree);
	
};

template<typename Ttree, typename Tvalue>
VoxelGrid<Ttree, Tvalue>::VoxelGrid() {}

template<typename Ttree, typename Tvalue>
VoxelGrid<Ttree, Tvalue>::~VoxelGrid() {}

template<typename Ttree, typename Tvalue>
void VoxelGrid<Ttree, Tvalue>::create(Ttree * tree, 
										const BoundingBox & b,
										int maxLevel,
										int minLevel)
{
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
				box.reset();
				tree->intersectBox(&box);
				if(box.numIntersect() > 0 )
					addCell(sample, level, box.numIntersect() );
            }
        }
    }
	
	sdb::CellHash cellsToRefine;
	bool needRefine = tagCellsToRefine(cellsToRefine);
    while(needRefine && level < m_maxLevel) {
        refine(tree, cellsToRefine);
		level++;
		if(level < m_maxLevel) needRefine = tagCellsToRefine(cellsToRefine);
		std::cout<<"\n refine level"<<level<<" n cell "<<numCells();
    }
	
	std::cout<<"\n reach level "<<level;
	
	level = 3;
	while(level < minLevel) {
		if(tagLevelCellsToRefine(cellsToRefine, level) ) {
			refine(tree, cellsToRefine);
			std::cout<<"\n refine level "<<level<<" n cell "<<numCells();
		}
		level++;
	}
		
	createVoxels(tree);
}

template<typename Ttree, typename Tvalue>
bool VoxelGrid<Ttree, Tvalue>::tagCellsToRefine(sdb::CellHash & cellsToRefine)
{
	cellsToRefine.clear();
	sdb::CellHash * c = cells();
    c->begin();
    while(!c->end()) {
        if(c->value()->visited > 3) {
			sdb::CellValue * ind = new sdb::CellValue;
			ind->level = c->value()->level;
			ind->visited = 1;
			cellsToRefine.insert(c->key(), ind);
		}
        c->next();
	}
	if(cellsToRefine.size() < 1 ) return false;
	
	tagCellsToRefineByNeighbours(cellsToRefine);
	tagCellsToRefineByNeighbours(cellsToRefine);
	tagCellsToRefineByNeighbours(cellsToRefine);
	tagCellsToRefineByNeighbours(cellsToRefine);
	
	return true;
}

template<typename Ttree, typename Tvalue>
bool VoxelGrid<Ttree, Tvalue>::tagLevelCellsToRefine(sdb::CellHash & cellsToRefine,
														const int & level)
{
	cellsToRefine.clear();
	sdb::CellHash * c = cells();
    c->begin();
    while(!c->end()) {
        if(c->value()->level == level) {
			sdb::CellValue * ind = new sdb::CellValue;
			ind->level = c->value()->level;
			ind->visited = 1;
			cellsToRefine.insert(c->key(), ind);
		}
        c->next();
	}
	return cellsToRefine.size() > 0;
}

template<typename Ttree, typename Tvalue>
void VoxelGrid<Ttree, Tvalue>::refine(Ttree * tree, sdb::CellHash & cellsToRefine)
{    
	int level1;
	float hh;
    Vector3F sample, subs;
	int u;
	unsigned k;
	BoxIntersectContext box;
	cellsToRefine.begin();
	while (!cellsToRefine.end()) {
		sdb::CellValue * parentCell = cellsToRefine.value();

		k = cellsToRefine.key();
		
		level1 = parentCell->level + 1;
		hh = cellSizeAtLevel(level1) * .5f;
		sample = cellCenter(k);
		removeCell(k);
		for(u = 0; u < 8; u++) {
			subs = sample + Vector3F(hh * Cell8ChildOffset[u][0], 
			hh * Cell8ChildOffset[u][1], 
			hh * Cell8ChildOffset[u][2]);
			box.setMin(subs.x - hh, subs.y - hh, subs.z - hh);
			box.setMax(subs.x + hh, subs.y + hh, subs.z + hh);
			box.reset();
			tree->intersectBox(&box);
			if(box.numIntersect() > 0) 
				addCell(subs, level1, box.numIntersect());
		}
		
		cellsToRefine.next();
    }
}

template<typename Ttree, typename Tvalue>
void VoxelGrid<Ttree, Tvalue>::createVoxels(Ttree * tree)
{
	int minPrims = 1<<20;
	int maxPrims = 0;
	int totalPrims = 0;
	int nCells = 0;
	int nPrims;
	float hh;
    Vector3F sample;
	BoxIntersectContext box;
	sdb::CellHash * c = cells();
    c->begin();
    while(!c->end()) {
	
		sample = cellCenter(c->key() );
		hh = cellSizeAtLevel(c->value()->level ) * 0.5f;
		
		box.setMin(sample.x - hh, sample.y - hh, sample.z - hh);
		box.setMax(sample.x + hh, sample.y + hh, sample.z + hh);
		box.reset();
        
		tree->intersectBox(&box);
		
		nPrims = box.numIntersect();
		if(nPrims < 1) std::cout<<"\n warning no intersect";
		if(minPrims > nPrims ) minPrims = nPrims;
		if(maxPrims < nPrims ) maxPrims = nPrims;
		nCells++;
		totalPrims += nPrims;
		
		Voxel v;
		v.setColor(.99f, .99f, .99f, .99f);
		v.setPos(encodeCellOrigin(c->key(), c->value()->level ), c->value()->level );
		v.setContour(0, 0);
		m_voxels.insert(v);
		
        c->next();
	}
	
	std::cout<<"\n n prims per cell min/max/average "<<minPrims
	<<" / "<<maxPrims<<" / "<<(float)totalPrims/(float)nCells
	<<"\n n voxel "<<numVoxels();
}

template<typename Ttree, typename Tvalue>
int VoxelGrid<Ttree, Tvalue>::numVoxels() const
{ return m_voxels.size(); }

template<typename Ttree, typename Tvalue>
int VoxelGrid<Ttree, Tvalue>::numContours() const
{ return m_contours.size(); }

}