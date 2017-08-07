/*
 *  Hilbert2D.h
 *  
 *
 *  Created by jian zhang on 6/1/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef TTG_HILBERT_2D_H
#define TTG_HILBERT_2D_H

#include <math/Vector3F.h>
#include <math/QuickSort.h>

namespace aphid {
class GeoDrawer;
}

class Hilbert2D {

	int m_level;
	int m_N;
	aphid::Vector3F * m_X;
	aphid::QuickSortPair<int, int> * m_ind;
	
public:
	Hilbert2D();
	virtual ~Hilbert2D();
	
	virtual bool init();
	virtual bool progressForward();
	virtual bool progressBackward();
	virtual void draw(aphid::GeoDrawer * dr);
	
	void printCoord();
	
private:
	void generateSamples(int level);
	
		
};

#endif