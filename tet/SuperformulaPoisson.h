/*
 *  SuperformulaPoisson.h
 *  
 *
 *  Created by jian zhang on 6/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef TTG_SUPERFORMULA_POISSON_H
#define TTG_SUPERFORMULA_POISSON_H
#include "SuperformulaTest.h"
#include "PoissonSequence.h"

namespace ttg {

struct Disk {

	aphid::Vector3F pos;
	float r;
	int key;
	
	bool collide(const Disk * b) const
	{
		return (pos.distanceTo(b->pos)
				< (r + b->r) );
	}
	
};

class SuperformulaPoisson : public SuperformulaTest {

public:
	SuperformulaPoisson();
	virtual ~SuperformulaPoisson();
	
	virtual const char * titleStr() const;
	
protected:
	virtual bool createSamples();
	
	
};

}
#endif