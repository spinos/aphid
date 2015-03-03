/*
 *  DrawNp.h
 *  testnarrowpahse
 *
 *  Created by jian zhang on 3/3/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
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
	void drawSeparateAxis(CudaNarrowphase * phase);
private:
	GeoDrawer * m_drawer;
	BaseBuffer * m_x1;
	BaseBuffer * m_separateAxis;
};