/*
 *  DrawNp.h
 *  testnarrowpahse
 *
 *  Created by jian zhang on 3/3/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <AllMath.h>
class GeoDrawer;
class TetrahedronSystem;
class BaseBuffer;
class CudaNarrowphase;
class SimpleContactSolver;
class BaseLog;
class DrawNp {
public:
	DrawNp();
	virtual ~DrawNp();
	
	void setDrawer(GeoDrawer * drawer);

	void drawTetra(TetrahedronSystem * tetra);
	void drawTetraAtFrameEnd(TetrahedronSystem * tetra);
	void drawSeparateAxis(CudaNarrowphase * phase, BaseBuffer * pairs, TetrahedronSystem * tetra);
	bool checkConstraint(SimpleContactSolver * solver, CudaNarrowphase * phase, TetrahedronSystem * tetra);
	void printCoord(CudaNarrowphase * phase, BaseBuffer * pairs);
	void printTOI(CudaNarrowphase * phase, BaseBuffer * pairs);
	void printContactPairHash(SimpleContactSolver * solver, unsigned numContacts);
private:
    void computeX1(TetrahedronSystem * tetra, float h = 0.0166667f);
    Vector3F tetrahedronCenter(Vector3F * p, unsigned * v, unsigned i);
    Vector3F tetrahedronVelocity(TetrahedronSystem * tetra, unsigned * v, unsigned i);
    Vector3F interpolatePointTetrahedron(Vector3F * p, unsigned * v, unsigned i, float * wei);
private:
	GeoDrawer * m_drawer;
	BaseBuffer * m_x1;
	BaseBuffer * m_coord;
	BaseBuffer * m_contact;
	BaseBuffer * m_counts;
	BaseBuffer * m_contactPairs;
	BaseBuffer * m_scanResult;
	BaseBuffer * m_pairsHash;
	BaseBuffer * m_linearVelocity;
	BaseBuffer * m_angularVelocity;
	BaseBuffer * m_deltaJ;
	BaseBuffer * m_pntTetHash;
	BaseBuffer * m_constraint;
	BaseLog * m_log;
};