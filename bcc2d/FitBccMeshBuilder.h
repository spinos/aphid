/*
 *  FitBccMeshBuilder.h
 *  testbcc
 *
 *  Created by jian zhang on 4/27/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <TetrahedronMeshBuilder.h>
class Vector3F;
class KdTreeDrawer;
class BezierCurve;
struct BezierSpline;
class BccOctahedron;
class KdTreeDrawer;
class CurveSampler;
class SampleGroup;

class FitBccMeshBuilder : public TetrahedronMeshBuilder {
public:
	FitBccMeshBuilder();
	virtual ~FitBccMeshBuilder();
	
	virtual void build(GeometryArray * geos,
				unsigned & ntet, unsigned & nvert, unsigned & nstripes);
	
	virtual void addAnchors(ATetrahedronMesh * mesh, unsigned n, KdIntersection * anchorMesh);
	
protected:
	void build(BezierCurve * curve,
			   unsigned curveIdx);
	void drawOctahedron(KdTreeDrawer * drawer);
	void drawSamples(KdTreeDrawer * drawer);
private:
    void cleanup();
	void drawOctahedron(KdTreeDrawer * drawer, BccOctahedron & octa);
	Vector3F * startPoints();
	unsigned * tetrahedronDrifts();
	
private:
	CurveSampler * m_sampler;
	SampleGroup * m_reducer;
	BccOctahedron * m_octa;
	Vector3F * m_startPoints;
	unsigned * m_tetraDrift;
};
