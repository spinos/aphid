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
class DrawNp {
public:
	DrawNp();
	virtual ~DrawNp();
	
	void setDrawer(GeoDrawer * drawer);

	void drawTetra(TetrahedronSystem * tetra);
	void drawTetraAtFrameEnd(TetrahedronSystem * tetra);
	void drawSeparateAxis(CudaNarrowphase * phase, BaseBuffer * pairs, TetrahedronSystem * tetra);
private:
    void computeX1(TetrahedronSystem * tetra);
    Vector3F tetrahedronCenter(Vector3F * p, unsigned * v, unsigned i);
private:
	GeoDrawer * m_drawer;
	BaseBuffer * m_x1;
	BaseBuffer * m_separateAxis;
	BaseBuffer * m_localA;
	BaseBuffer * m_localB;
};