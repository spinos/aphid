/*
 *  FitTest.h
 *  testbcc
 *
 *  Created by jian zhang on 4/27/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <vector>
class Vector3F;
class KdTreeDrawer;
class BezierCurve;
struct BezierSpline;
class BccOctahedron;

class FitTest {
public:
	FitTest(KdTreeDrawer * drawer);
	virtual ~FitTest();
	
	void draw();
protected:

private:
	float splineLength(BezierSpline & spline);
	float splineParameterByLength(BezierSpline & spline, float expectedLength);
	void drawOctahedron(BccOctahedron & octa);
	void drawTetrahedron();
private:
	KdTreeDrawer * m_drawer;
	BezierCurve * m_curve;
	Vector3F * m_samples;
	unsigned m_numSamples;
	Vector3F * m_reducedP;
	unsigned m_numGroups;
	float * m_octahedronSize;
	BccOctahedron * m_octa;
	std::vector<Vector3F > m_tetrahedronP;
	std::vector<unsigned > m_tetrahedronInd;
};
