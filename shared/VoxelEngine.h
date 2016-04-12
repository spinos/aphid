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

struct Contour {
/// point-normal packed into int
/// bit layout
/// 0-3		gridx		4bit
/// 4-7		girdy		4bit
/// 8-11	girdz		4bit
/// 12-14   thickness   3bit slab if > 0
/// 15 normal sign		1bit
/// 16-17 normal axis	2bit
/// 18-23 normal u		6bit
/// 24-29 normal v		6bit
	int m_data;
	
	void reset()
	{ m_data = 0; }
	
	void setNormal(const Vector3F & n)
	{ colnor30::encodeN(m_data, n); }
	
	void setThickness(const float & x,
						const float & d)
	{
		int ng = (x / d) * 8.f;
		if(ng > 7) ng = 7;
		m_data = m_data | (ng<<12); 
	}
	
	void setPoint(const Vector3F & p,
					const Vector3F & o,
					const float & d)
	{
		Vector3F c((p.x - o.x)/d, (p.y - o.y)/d, (p.z - o.z)/d);
		col12::encodeC(m_data, c);
	}
	
	Vector3F getPoint(const Vector3F & o,
					const float & d) const
	{
		Vector3F r;
		col12::decodeC(r, m_data);
		r.x = d * r.x + o.x;
		r.y = d * r.y + o.y;
		r.z = d * r.z + o.z;
		return r;
	}
	
	Vector3F getNormal() const
	{ 
		Vector3F r;
		colnor30::decodeN(r, m_data);
		return r;
	}
	
	float getThickness(const float & d) const
	{
		int ng = (m_data & 32767)>>12;
		return (float)ng * d * .125f;
	}
	
};

struct Voxel {
/// morton code of cell center in bound
	int m_pos;
/// level-ncontour packed into int
/// bit layout
/// 0-3 level			4bit 0-15
/// 4-6 n contour		3bit 0-7
	int m_level;
/// color in rgba 32bit
	int m_color;
	Contour m_contour[7];
	
	void setColor(const float &r, const float &g,
					const float &b, const float &a)
	{ col32::encodeC(m_color, r, g, b, a); }
	
/// set pos first
	void setPos(const int & morton, const int & level)
	{
		m_pos = morton; 
		m_level = level;
	}
	
	void setNContour(const int & count)
	{ m_level = m_level | (count<<4); }
	
	void getColor(float &r, float &g,
					float &b, float &a)
	{ col32::decodeC(r, g, b, a, m_color); }
	
	BoundingBox calculateBBox() const
	{
		unsigned x, y, z;
		decodeMorton3D(m_pos, x, y, z);
		float h = 1<< (9 - (m_level & 15) );
		
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
	
	int getNContour() const
	{ return (m_level & 255) >> 4; }
	
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
	
	void extractColor(Voxel & dst) const;
	void extractContours(Voxel & dst) const;
	void printContours(const Voxel & v) const;
	const AOrientedBox & orientedBBox() const;
	
protected:
	void calculateOBox(Profile * prof);
	void sampleCells(std::vector<Vector3F> & dst);
	void samplePrims(std::vector<Vector3F> & dst, Profile * prof);

private:
	bool findIntersection(Vector3F & dst,
						const Vector3F & dir,
						const BoundingBox & box,
						const float & d) const;

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
	m_obox.limitMinThickness(cellSizeAtLevel(NLevel + 1) );
}

template<typename T, typename Tn, int NLevel>
const AOrientedBox & VoxelEngine<T, Tn, NLevel>::orientedBBox() const
{ return m_obox; }

template<typename T, typename Tn, int NLevel>
void VoxelEngine<T, Tn, NLevel>::extractColor(Voxel & dst) const
{
	dst.setColor(.99f, .99f, .99f, .99f);
}

template<typename T, typename Tn, int NLevel>
void VoxelEngine<T, Tn, NLevel>::extractContours(Voxel & dst) const
{
#define VERBEXTRACTCONTOUR 1
	int nct = 0;
	BoundingBox box;
	getBounding(box);
	const Vector3F & cnt = m_obox.center();
	const Matrix33F & rot = m_obox.orientation();
	const Vector3F & oboxExt = m_obox.extent();
	const Vector3F & ori = box.getMin();
	const float d = box.distance(0);
/// reference of thickness
	const float hd = d * .866f;
	const Vector3F facex = m_obox.get8DOPFaceX();
	const Vector3F facey = m_obox.get8DOPFaceY();
#if VERBEXTRACTCONTOUR
	std::cout<<"\n voxel bound"<<box;
	std::cout<<"\n dop facing "<<facex<<" "<<facey
					<<"\n extent "<<m_obox.dopExtent()[0]<<" "<<m_obox.dopExtent()[1]
					<<"\n        "<<m_obox.dopExtent()[2]<<" "<<m_obox.dopExtent()[3];
#endif
	
/// 3 slabs of oriented box
	Contour acontour;
	acontour.reset();
	acontour.setPoint(cnt, ori, d);
	acontour.setThickness(oboxExt.x, hd);
	acontour.setNormal(rot.row(0) );
#if VERBEXTRACTCONTOUR
	std::cout<<"\n slab 0 at "<<cnt<<"\n facing "<<rot.row(0)
							<<"\n thickness "<<oboxExt.x;
#endif
	
	dst.m_contour[nct++] = acontour;
	
	acontour.reset();
	acontour.setPoint(cnt, ori, d);
	acontour.setThickness(oboxExt.y, hd);
	acontour.setNormal(rot.row(1) );
#if VERBEXTRACTCONTOUR
	std::cout<<"\n slab 1 at "<<cnt<<"\n facing "<<rot.row(1)
							<<"\n thickness "<<oboxExt.y;
#endif	
	dst.m_contour[nct++] = acontour;
	
	acontour.reset();
	acontour.setPoint(cnt, ori, d);
	acontour.setThickness(oboxExt.z, hd);
	acontour.setNormal(rot.row(2) );
#if VERBEXTRACTCONTOUR
	std::cout<<"\n slab 2 at "<<cnt<<"\n facing "<<rot.row(2)
							<<"\n thickness "<<oboxExt.z;
#endif	
	dst.m_contour[nct++] = acontour;
	
/// 4 dop cuts
	Vector3F cutp = cnt + facex * m_obox.dopExtent()[0];
	Vector3F cutn = facex * -1.f;
	Vector3F cutd = facey * -1.f;
	bool hasContour = box.isPointInside(cutp);
	if(!hasContour) {
#if VERBEXTRACTCONTOUR
	std::cout<<"\n cut 0 "<<cutp<<" out of box ";
#endif
		hasContour = findIntersection(cutp, cutd, box, d);
	}
	
	if(hasContour) {
		acontour.reset();
		acontour.setPoint(cutp, ori, d);
		acontour.setNormal(cutn);
		dst.m_contour[nct++] = acontour;
#if VERBEXTRACTCONTOUR
	std::cout<<"\n cut 0 at "<<cutp<<"\n facing "<<cutn;
#endif
	}
	
	cutp = cnt + facex * m_obox.dopExtent()[1];
	cutn = facex;
	cutd = facey;
	hasContour = box.isPointInside(cutp);
	if(!hasContour) {
#if VERBEXTRACTCONTOUR
	std::cout<<"\n cut 1 "<<cutp<<" out of box ";
#endif
		hasContour = findIntersection(cutp, cutd, box, d);
	}
	
	if(hasContour) {
		acontour.reset();
		acontour.setPoint(cutp, ori, d);
		acontour.setNormal(cutn);
		dst.m_contour[nct++] = acontour;
#if VERBEXTRACTCONTOUR
	std::cout<<"\n cut 1 at "<<cutp<<"\n facing "<<cutn;
#endif
	}
	
	cutp = cnt + facey * m_obox.dopExtent()[2];
	cutn = facey * -1.f;
	cutd = facex;
	hasContour = box.isPointInside(cutp);
	if(!hasContour) {
#if VERBEXTRACTCONTOUR
	std::cout<<"\n cut 2 "<<cutp<<" out of box ";
#endif
		hasContour = findIntersection(cutp, cutd, box, d);
	}
	
	if(hasContour) {
		acontour.reset();
		acontour.setPoint(cutp, ori, d);
		acontour.setNormal(cutn);
		dst.m_contour[nct++] = acontour;
#if VERBEXTRACTCONTOUR
	std::cout<<"\n cut 2 at "<<cutp<<"\n facing "<<cutn;
#endif
	}
	
	cutp = cnt + facey * m_obox.dopExtent()[3];
	cutn = facey;
	cutd = facex * -1.f;
	hasContour = box.isPointInside(cutp);
	if(!hasContour) {
#if VERBEXTRACTCONTOUR
	std::cout<<"\n cut 3 "<<cutp<<" out of box ";
#endif
		hasContour = findIntersection(cutp, cutd, box, d);
	}
	
	if(hasContour) {
		acontour.reset();
		acontour.setPoint(cutp, ori, d);
		acontour.setNormal(cutn);
		dst.m_contour[nct++] = acontour;
#if VERBEXTRACTCONTOUR
	std::cout<<"\n cut 3 at "<<cutp<<"\n facing "<<cutn;
#endif
	}
#if VERBEXTRACTCONTOUR
	std::cout<<"\n extract "<<nct<<" contours ";
#endif	
	dst.setNContour(nct);
	std::cout.flush();
}

template<typename T, typename Tn, int NLevel>
bool VoxelEngine<T, Tn, NLevel>::findIntersection(Vector3F & dst,
										const Vector3F & dir,
										const BoundingBox & box,
										const float & d) const
{
	float tmin, tmax;
	Ray incident(dst, dir, 0.f, d);
	bool stat = box.intersect(incident, &tmin, &tmax);
	if(!stat) {
		incident = Ray(dst, dir.reversed(), 0.f, d);
		stat = box.intersect(incident, &tmin, &tmax);
	}
/// put p inside
	if(stat) dst = incident.travel(tmin + 1e-5f);
	return stat;
}

template<typename T, typename Tn, int NLevel>
void VoxelEngine<T, Tn, NLevel>::printContours(const Voxel & v) const
{
	const Vector3F & boxOri = origin();
	const float & d = span(); 
	const int nct = v.getNContour();
	int i=0;
	for(;i<nct;++i) {
		const Contour & c = v.m_contour[i];
		std::cout<<"\n contour["<<i<<"] at "<<c.getPoint(boxOri, d)
			<<"\n facing "<<c.getNormal()
			<<"\n thickness "<<c.getThickness(d * .866f);
	}
	std::cout<<"\n voxel has "<<nct<<" contours";
	std::cout.flush();
}

}

#endif
