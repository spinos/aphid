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
#include <Morton3D.h>
#include <PrincipalComponents.h>
#include <Quantization.h>
#include <vector>

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
		b.expand(-.000023f);
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

template<typename T, typename Tn, int NLevel = 3>
class VoxelEngine : public CartesianGrid {

	AOrientedBox m_obox;
	
public:
	struct Profile {
		KdNTree<T, Tn > * _tree;
		bool _approxCell;
		bool _orientAtXY;
		Profile() {
			_tree = NULL;
			_approxCell = false;
			_orientAtXY = false;
		}
	};

	VoxelEngine();
	virtual ~VoxelEngine();
	
	bool build(Profile * prof);
	
	void extractContours(Voxel & dst) const;
	
/// access to primitives
	const std::vector<T> & prims() const;
	const AOrientedBox & orientedBBox() const;
	
protected:
	void calculateOBox(Profile * prof);
	void sampleCells(std::vector<Vector3F> & dst);
	void samplePrims(std::vector<Vector3F> & dst, Profile * prof);

private:

};

template<typename T, typename Tn, int NLevel>
VoxelEngine<T, Tn, NLevel>::VoxelEngine()
{}

template<typename T, typename Tn, int NLevel>
VoxelEngine<T, Tn, NLevel>::~VoxelEngine()
{}

template<typename T, typename Tn, int NLevel>
bool VoxelEngine<T, Tn, NLevel>::build(Profile * prof)
{
	const float h = cellSizeAtLevel(NLevel);
    const float hh = h * .49995f;
	const int dim = 1<<NLevel;
	const Vector3F ori = origin() + Vector3F(hh, hh, hh);
    Vector3F sample;
	KdEngine eng;
	BoxIntersectContext box;
	int i, j, k;
	for(k=0; k < dim; k++) {
        for(j=0; j < dim; j++) {
            for(i=0; i < dim; i++) {
                sample = ori + Vector3F(h* (float)i, h* (float)j, h* (float)k);
                box.setMin(sample.x - hh, sample.y - hh, sample.z - hh);
                box.setMax(sample.x + hh, sample.y + hh, sample.z + hh);
				box.reset(1, true);
				
				eng.intersectBox<T, Tn>(prof->_tree, &box);
				if(box.numIntersect() > 0 )
					addCell(sample, NLevel, 1);
            }
        }
    }
	
	if(numCells() < 1) return false;
	
	calculateOBox(prof);
	
	return true;
}

template<typename T, typename Tn, int NLevel>
void VoxelEngine<T, Tn, NLevel>::sampleCells(std::vector<Vector3F> & dst)
{
	const float hh = cellSizeAtLevel(NLevel + 2);
	sdb::CellHash * c = cells();
    c->begin();
    while(!c->end()) {
        
		Vector3F cnt = cellCenter(c->key() );
		for(int i=0; i< 8; ++i)
			dst.push_back(cnt + Vector3F(Cell8ChildOffset[i][0],
											Cell8ChildOffset[i][1],
											Cell8ChildOffset[i][2]) * hh );
			
        c->next();
	}
}

template<typename T, typename Tn, int NLevel>
void VoxelEngine<T, Tn, NLevel>::samplePrims(std::vector<Vector3F> & dst, Profile * prof)
{
	BoxIntersectContext box;
	getBounding(box);
	box.reset(1<<16, true);
	KdEngine eng;
	eng.intersectBox<T, Tn>(prof->_tree, &box);
	
	const std::vector<int> & inds = box.primIndices();
	const sdb::VectorArray<T> * prims = prof->_tree->source();
    
	Vector3F p;
	
	typename std::vector<int>::const_iterator it;
	for(int i=0; i< 200; ++i) {
		it = inds.begin();
		for(;it!= inds.end(); ++it) {
			const T * aprim = prims->get(*it); 
			if(aprim->sampleP(p, box) )
					dst.push_back(p);
		}
		if(dst.size() > 500) return;
	}
}

template<typename T, typename Tn, int NLevel>
void VoxelEngine<T, Tn, NLevel>::calculateOBox(Profile * prof)
{
	std::vector<Vector3F> pnts;
	
	if(prof->_approxCell) sampleCells(pnts);
	else {
		samplePrims(pnts, prof);
		if(pnts.size() < 32) sampleCells(pnts);
	}
	
	PrincipalComponents<std::vector<Vector3F> > obpca;
	if(prof->_orientAtXY) obpca.setOrientConstrain(1);
	m_obox = obpca.analyze(pnts, pnts.size() );
	m_obox.limitMinThickness(cellSizeAtLevel(NLevel + 2) );
}

template<typename T, typename Tn, int NLevel>
const AOrientedBox & VoxelEngine<T, Tn, NLevel>::orientedBBox() const
{ return m_obox; }

template<typename T, typename Tn, int NLevel>
void VoxelEngine<T, Tn, NLevel>::extractContours(Voxel & dst) const
{
}

}

#endif
