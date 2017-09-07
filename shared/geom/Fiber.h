/*
 *  Fiber.h
 *  
 *
 *  Created by jian zhang on 9/8/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_FIBER_H
#define APH_FIBER_H

#include <math/Quaternion.h>
#include <math/Vector3F.h>
#include <boost/scoped_ptr.hpp>
#include <boost/scoped_array.hpp>
#include <vector>
#include <deque>

namespace aphid {

class Ray;
class Matrix44F;
class SegmentNormals;

/// rotation and position
struct OrientedPoint {	
	Quaternion _q;
	Quaternion _q0i;
	Vector3F _x;
	Vector3F _x0;
};

struct FiberUnit {
	Vector3F _tng0;
	Vector3F _tng1;
	OrientedPoint* _v0;
	OrientedPoint* _v1;
};


/// a single strand of
class FiberBulder {

	std::deque<Vector3F > m_pnts;
	boost::scoped_ptr<SegmentNormals > m_normals;
	
public:
	FiberBulder();
	virtual ~FiberBulder();
	
/// clear points
	void begin();
	void addPoint(const Vector3F& pv);
	void setPoint(int i, const Vector3F& pv);
/// after i-th point
	void insertPoint(int i, const Vector3F& pv);
/// calculate segment normals
	void end();
	
	int numPoints() const;
	int numSegments() const;
/// i: [0,np-1]
	const Vector3F& getPoint(int i) const;
/// i-th segment normal i: [0,np-2] 
	const Vector3F& getNormal(int i) const;
	
	void copyStrand(OrientedPoint* pnts,
				FiberUnit* segs) const;
	
protected:

private:
/// q of i-th point
	void setRotation(OrientedPoint* dst,
				const Vector3F& vx, 
				const Vector3F& vy, const int& i) const;
};

class Fiber {

	OrientedPoint* m_pnts;
	FiberUnit* m_segs;
	int m_numPnts;
	
public:
	Fiber();
	virtual ~Fiber();
	
	void create(const FiberBulder* bld,
			OrientedPoint* pnt,
			FiberUnit* seg);
	void dump(FiberBulder* bld) const;
	void initialize();

	const int& numPoints() const;
	int numSegments() const;
	
	OrientedPoint* points();
	FiberUnit* segments();
	
	const OrientedPoint* points() const;
	const FiberUnit* segments() const;
	
/// segments
	void update();
	
	static void InterpolateSpace(Matrix44F& mat, 
					float& segl,
					const FiberUnit& fu,
					const float& t);
					
	static void InterpolatePoint(Vector3F& pv,
					const FiberUnit& fu,
					const float& t);
	
protected:
/// x and x0 of i-th point
	void setPoint(const Vector3F& pv, const int& i);
/// q of i-th point
	void setRotation(const Vector3F& vx, 
				const Vector3F& vy, const int& i);
				
private:
/// up of i-th point
	Vector3F getPointUp(const int& i) const;
	
};

}

#endif
