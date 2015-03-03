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
class DrawNp {
public:
	DrawNp();
	virtual ~DrawNp();
	
	void setDrawer(GeoDrawer * drawer);

	void drawTetra(TetrahedronSystem * tetra);
	void drawTetraAtFrameEnd(TetrahedronSystem * tetra);
private:
	GeoDrawer * m_drawer;
	BaseBuffer * m_x1;
};