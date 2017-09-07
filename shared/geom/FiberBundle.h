/*
 *  FiberBundle.h
 *  
 *
 *  Created by jian zhang on 9/9/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_FIBER_BUNDLE_H
#define APH_FIBER_BUNDLE_H

#include "Fiber.h"

namespace aphid {

/// multiple strands of
class FiberBundleBuilder {

	std::deque<FiberBulder* > m_strands;
	
public:
	FiberBundleBuilder();
	virtual ~FiberBundleBuilder();
	
	int numStrands() const;
	void addStrand();
	void addPoint(const Vector3F& pv);
	void clear();
	void begin();
	void end();
/// after i-th point of j-th strand
	void insertPoint(int i, int j,
				const Vector3F& pv);
	
	const FiberBulder* strand(int i) const;
	FiberBulder* strand(int i);
	
protected:
	FiberBulder* lastStrand();
	
private:
};


class FiberBundle {
/// np # points ns # segments
	boost::scoped_array<OrientedPoint > m_pnts;
	boost::scoped_array<FiberUnit > m_segs;
/// # strand + 1 strand begins
/// ind to point
	boost::scoped_array<int > m_strandBegins;
	boost::scoped_array<Fiber > m_fibers;
	int m_numStrands;
	int m_numPnts;
	int m_numSegs;
	int m_selectedStrand, m_selectedPoint;
	
public:
	FiberBundle();
	virtual ~FiberBundle();
	
	void create(const FiberBundleBuilder& bld);
	void dump(FiberBundleBuilder* bld) const;
	void initialize();
	void update();
	
	const int& numPoints() const;
	const int& numSegments() const;
	const int& numStrands() const;
	
	const OrientedPoint* points() const;
	const Fiber* strand(int i) const;
	
	bool selectPoint(float& minD, const Ray* incident);
	void getSelectPointSpace(Matrix44F* tm) const;
	
	void moveSelectedPoint(const Vector3F& dv);
	void rotateSelectedPoint(const Quaternion& dq);
/// until end of selected strand
	void moveSelectedStrand(const Vector3F& dv);
	void rotateSelectedStrand(const Quaternion& dq);
	
protected:

private:
};

}

#endif