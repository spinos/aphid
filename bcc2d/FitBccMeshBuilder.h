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
class KdTreeDrawer;
class CurveSampler;
class SampleGroup;
class FitBccMeshBuilder {
public:
	FitBccMeshBuilder();
	virtual ~FitBccMeshBuilder();
	
	void build(GeometryArray * curves, 
	           std::vector<Vector3F > & tetrahedronP, 
	           std::vector<unsigned > & tetrahedronInd);
	
	void build(BezierCurve * curve, 
	           std::vector<Vector3F > & tetrahedronP, 
	           std::vector<unsigned > & tetrahedronInd,
			   unsigned curveIdx);
			   
	Vector3F * startPoints();
	unsigned * tetrahedronDrifts();
			   
	static float EstimatedGroupSize;
protected:
	void drawOctahedron(KdTreeDrawer * drawer);
	void drawSamples(KdTreeDrawer * drawer);
private:
    void cleanup();
	void drawOctahedron(KdTreeDrawer * drawer, BccOctahedron & octa);
	void checkTetrahedronVolume(std::vector<Vector3F > & tetrahedronP, 
	           std::vector<unsigned > & tetrahedronInd, unsigned start);
private:
	CurveSampler * m_sampler;
	SampleGroup * m_reducer;
	BccOctahedron * m_octa;
	Vector3F * m_startPoints;
	unsigned * m_tetraDrift;
};
