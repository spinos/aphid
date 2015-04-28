/*
 *  FitBccMeshBuilder.h
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
class GeometryArray;

class FitBccMeshBuilder {
public:
	FitBccMeshBuilder();
	virtual ~FitBccMeshBuilder();
	
	void build(GeometryArray * curves, 
	           std::vector<Vector3F > & tetrahedronP, 
	           std::vector<unsigned > & tetrahedronInd,
	           float groupNCvRatio,
	           unsigned minNumGroups,
	           unsigned maxNumGroups);
	
	void build(BezierCurve * curve, 
	           std::vector<Vector3F > & tetrahedronP, 
	           std::vector<unsigned > & tetrahedronInd,
	           float groupNCvRatio,
	           unsigned minNumGroups,
	           unsigned maxNumGroups);
protected:
private:
    void cleanup();
	float splineLength(BezierSpline & spline);
	float splineParameterByLength(BezierSpline & spline, float expectedLength);
	
private:
	Vector3F * m_samples;
	unsigned m_numSamples;
	Vector3F * m_reducedP;
	unsigned m_numGroups;
	float * m_octahedronSize;
	BccOctahedron * m_octa;
};
