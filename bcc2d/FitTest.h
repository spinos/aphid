/*
 *  FitTest.h
 *  testbcc
 *
 *  Created by jian zhang on 4/27/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include "FitBccMeshBuilder.h"
class Vector3F;
class KdTreeDrawer;
class BezierCurve;
struct BezierSpline;
class BccOctahedron;
class GeometryArray;
class FitTest : public FitBccMeshBuilder {
public:
	FitTest(KdTreeDrawer * drawer);
	virtual ~FitTest();
	
	void draw();
protected:

private:
    void createSingleCurve();
    void createRandomCurves();
	void drawTetrahedron();
private:
	KdTreeDrawer * m_drawer;
	GeometryArray * m_allGeo;
	std::vector<Vector3F > m_tetrahedronP;
	std::vector<unsigned > m_tetrahedronInd;
    std::vector<unsigned > m_pdrift;
    std::vector<unsigned > m_idrift;
};
