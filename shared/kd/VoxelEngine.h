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
#include <Boundary.h>
#include <Morton3D.h>
#include <Quantization.h>
#include <kd/PrincipalComponents.h> 
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
		float fg = (x / d) * 8.f;
		int ng = fg;
		if(fg - ng > .5f) ng++;
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

template<typename T>
class PrimSampler {

public:
	PrimSampler();
	
	void run(std::vector<Vector3F> * dst, 
						const sdb::VectorArray<T> * prims,
						const BoundingBox & box,
						const std::vector<int> & inds,
						int minInd, int maxInd) const;
						
};

template<typename T>
PrimSampler<T>::PrimSampler()
{}

template<typename T>
void PrimSampler<T>::run(std::vector<Vector3F> * dst, 
						const sdb::VectorArray<T> * prims,
						const BoundingBox & box,
						const std::vector<int> & inds,
						int minInd, int maxInd) const
{
	Vector3F p;
	for(int j=0; j< 400; ++j) {
		
		for(int i=minInd; i<maxInd; ++i) {
			const T * aprim = prims->get(inds[i]); 
			if(aprim->sampleP(p, box) )
					dst->push_back(p);
		}
		if(dst->size() > 200) return;
	}
}

template<typename T, typename Tn, int NThread = 8>
class VoxelEngine : public Boundary {

	AOrientedBox m_obox;
	
public:
	struct Profile {
		BoundingBox _bbox;
		KdNTree<T, Tn > * _tree;
		bool _orientAtXY;
		Profile() {
			_tree = NULL;
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
	void samplePrims(std::vector<Vector3F> & dst, Profile * prof);

private:
	bool findIntersection(Vector3F & dst,
						const Vector3F & dir,
						const BoundingBox & box,
						const float & d) const;
	
};

template<typename T, typename Tn, int NThread>
VoxelEngine<T, Tn, NThread>::VoxelEngine()
{}

template<typename T, typename Tn, int NThread>
VoxelEngine<T, Tn, NThread>::~VoxelEngine()
{}

template<typename T, typename Tn, int NThread>
bool VoxelEngine<T, Tn, NThread>::build(Profile * prof)
{
	setBBox(prof->_bbox);
	std::vector<Vector3F> pnts;
	
	samplePrims(pnts, prof);
	if(pnts.size() < 8) {
		// std::cout<<"\n low n sample "<<pnts.size();
		return false;
	}
	
	PrincipalComponents<std::vector<Vector3F> > obpca;
	if(prof->_orientAtXY) obpca.setOrientConstrain(1);
	m_obox = obpca.analyze(pnts, pnts.size() );
	m_obox.limitMinThickness(prof->_bbox.distance(0) * .125f );
	
	return true;
}

template<typename T, typename Tn, int NThread>
void VoxelEngine<T, Tn, NThread>::samplePrims(std::vector<Vector3F> & dst, Profile * prof)
{
	BoxIntersectContext box(getBBox() );
	
	box.reset(1<<16, true);
	KdEngine eng;
	eng.intersectBox<T, Tn>(prof->_tree, &box);
	
	const std::vector<int> & inds = box.primIndices();
	const sdb::VectorArray<T> * prims = prof->_tree->source();
	
	const int nworks = inds.size();
#if 0 
	PrimSampler<T> sampler;
	sampler.run(&dst, prims, box, inds, 0, nworks );
#else	
	if(nworks < NThread) {
		PrimSampler<T> sampler;
		sampler.run(&dst, prims, box, inds, 0, nworks );
	}
	else {
		std::vector<Vector3F> branchSamples[NThread];
		boost::thread workTr[NThread];
		PrimSampler<T> sampler[NThread];
		
		const int wpt = nworks / NThread;
		int workMin, workMax;
/// branch
		for(int i=0; i<NThread; ++i) {
			workMin = i * wpt;
			workMax = workMin + wpt;
			if(workMax > nworks) workMax = nworks; 
			
			workTr[i] = boost::thread(boost::bind(&PrimSampler<T>::run, 
										&sampler[i], 
										&branchSamples[i], prims, box, inds, workMin, workMax) );
		}
/// join
		for(int i=0; i<NThread; ++i) {
			workTr[i].join();
		}
/// merge		
		for(int i=0; i<NThread; ++i) {
			const int branchSize = branchSamples[i].size();
			for(int j=0; j<branchSize; ++j) {
				dst.push_back(branchSamples[i][j]);
			}
		}
	}
#endif
}

template<typename T, typename Tn, int NThread>
const AOrientedBox & VoxelEngine<T, Tn, NThread>::orientedBBox() const
{ return m_obox; }

template<typename T, typename Tn, int NThread>
void VoxelEngine<T, Tn, NThread>::extractColor(Voxel & dst) const
{
/// todo
	dst.setColor(.99f, .99f, .99f, .99f);
}

template<typename T, typename Tn, int NThread>
void VoxelEngine<T, Tn, NThread>::extractContours(Voxel & dst) const
{
#define VERBEXTRACTCONTOUR 0
	int nct = 0;
	const BoundingBox & box = getBBox();
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
	std::cout<<"\n contour["<<nct<<"] at "<<cnt<<" facing "<<rot.row(0)
							<<" thickness "<<oboxExt.x;
#endif
	
	dst.m_contour[nct++] = acontour;
	
	acontour.reset();
	acontour.setPoint(cnt, ori, d);
	acontour.setThickness(oboxExt.y, hd);
	acontour.setNormal(rot.row(1) );
#if VERBEXTRACTCONTOUR
	std::cout<<"\n contour["<<nct<<"] at "<<cnt<<" facing "<<rot.row(1)
							<<" thickness "<<oboxExt.y;
#endif	
	dst.m_contour[nct++] = acontour;
	
	acontour.reset();
	acontour.setPoint(cnt, ori, d);
	acontour.setThickness(oboxExt.z, hd);
	acontour.setNormal(rot.row(2) );
#if VERBEXTRACTCONTOUR
	std::cout<<"\n contour["<<nct<<"] at "<<cnt<<" facing "<<rot.row(2)
							<<" thickness "<<oboxExt.z;
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
	std::cout<<"\n contour["<<nct<<"] at "<<cutp<<" facing "<<cutn;
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
	std::cout<<"\n contour["<<nct<<"] at "<<cutp<<" facing "<<cutn;
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
	std::cout<<"\n contour["<<nct<<"] at "<<cutp<<" facing "<<cutn;
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
	std::cout<<"\n contour["<<nct<<"] at "<<cutp<<" facing "<<cutn;
#endif
	}
#if VERBEXTRACTCONTOUR
	std::cout<<"\n extract "<<nct<<" contours ";
#endif	
	dst.setNContour(nct);
	std::cout.flush();
}

template<typename T, typename Tn, int NThread>
bool VoxelEngine<T, Tn, NThread>::findIntersection(Vector3F & dst,
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
	if(stat) dst = incident.travel(tmin + 1e-3f);
	return stat;
}

template<typename T, typename Tn, int NThread>
void VoxelEngine<T, Tn, NThread>::printContours(const Voxel & v) const
{
	const Vector3F & boxOri = getBBox().getMin();
	const float & d =  getBBox().distance(0); 
	const int nct = v.getNContour();
	int i=0;
	for(;i<nct;++i) {
		const Contour & c = v.m_contour[i];
		std::cout<<"\n contour["<<i<<"] at "<<c.getPoint(boxOri, d)
			<<" facing "<<c.getNormal()
			<<" thickness "<<c.getThickness(d * .866f);
	}
	std::cout<<"\n voxel has "<<nct<<" contours";
	std::cout.flush();
}

}

#endif
