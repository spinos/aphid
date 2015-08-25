/*
 *  FitBccMeshBuilder.h
 *  testbcc
 *
 *  Created by jian zhang on 4/27/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
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
class ATetrahedronMesh;
class ATetrahedronMeshGroup;
class KdIntersection;
class FitBccMeshBuilder {
public:
	FitBccMeshBuilder();
	virtual ~FitBccMeshBuilder();
	
	void build(GeometryArray * curves,
				unsigned & ntet, unsigned & nvert, unsigned & nstripes);
	
	void build(BezierCurve * curve,
			   unsigned curveIdx);
			   
	void getResult(ATetrahedronMeshGroup * m);
			   
	Vector3F * startPoints();
	unsigned * tetrahedronDrifts();
	
	void addAnchors(ATetrahedronMesh * mesh, unsigned n, KdIntersection * anchorMesh);
			   
	static float EstimatedGroupSize;
protected:
	void drawOctahedron(KdTreeDrawer * drawer);
	void drawSamples(KdTreeDrawer * drawer);
private:
    void cleanup();
	void drawOctahedron(KdTreeDrawer * drawer, BccOctahedron & octa);
	
private:
	std::vector<Vector3F > tetrahedronP;
	std::vector<unsigned > tetrahedronInd;
    std::vector<unsigned > pointDrifts;
    std::vector<unsigned > indexDrifts;
	
	CurveSampler * m_sampler;
	SampleGroup * m_reducer;
	BccOctahedron * m_octa;
	Vector3F * m_startPoints;
	unsigned * m_tetraDrift;
};
