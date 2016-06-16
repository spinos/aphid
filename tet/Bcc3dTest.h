/*
 *  Bcc3dTest.h
 *  
 *
 *  Created by jian zhang on 6/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef TTG_BCC_3D_H
#define TTG_BCC_3D_H
#include "Scene.h"
#include "BccTetraGrid.h"

namespace ttg {

class Bcc3dTest : public Scene {

	BccTetraGrid m_grid;
	aphid::Vector3F * m_X;
	int m_N;
	std::vector<ITetrahedron *> m_tets;
	
public:
	Bcc3dTest();
	virtual ~Bcc3dTest();
	
	virtual const char * titleStr() const;
	virtual bool init();
	virtual bool progressForward();
	virtual bool progressBackward();
	virtual void draw(aphid::GeoDrawer * dr);
	
private:
	void createGrid();
	
};

}
#endif