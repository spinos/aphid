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

	PoissonSequence<Disk> m_bkg;
	
public:
	SuperformulaPoisson();
	virtual ~SuperformulaPoisson();
	
	virtual const char * titleStr() const;
	virtual void draw(aphid::GeoDrawer * dr);
	
protected:
	virtual bool createSamples();
	
	PoissonSequence<Disk> * sampleGrid();
	
	void extractSamplePos(aphid::Vector3F * dst);
	
private:
	void fillBackgroud(PoissonSequence<Disk> * dst,
						PoissonSequence<Disk> * frontGrid,
						int & n);
	bool fillBackgroudAt(PoissonSequence<Disk> * dst,
						PoissonSequence<Disk> * frontGrid,
						Disk & cand,
						int & n);
	void drawSamplesIn(aphid::GeoDrawer * dr,
						aphid::sdb::Array<int, Disk > * cell);
						
	void extractPos(aphid::Vector3F * dst, PoissonSequence<Disk> * grid);
	void extractPosIn(aphid::Vector3F * dst, int & count,
					aphid::sdb::Array<int, Disk > * cell);
	
};

}
#endif