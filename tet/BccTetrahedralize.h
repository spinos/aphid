/*
 *  BccTetrahedralize.h
 *  
 *
 *  Created by jian zhang on 6/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef TTG_BCC_TETRAHEDRALIZE_H
#define TTG_BCC_TETRAHEDRALIZE_H
#include "SuperformulaPoisson.h"
#include "BccTetraGrid.h"

namespace ttg {

class BccTetrahedralize : public SuperformulaPoisson {

	BccTetraGrid m_grid;
	std::vector<ITetrahedron *> m_tets;
	int m_N, m_sampleBegin;
	float m_pntSz;
	aphid::Vector3F * m_X;
	
public:
	BccTetrahedralize();
	virtual ~BccTetrahedralize();
	
	virtual const char * titleStr() const;
	virtual bool createSamples();
	virtual void draw(aphid::GeoDrawer * dr);
	
protected:
	
private:
	
};

}
#endif