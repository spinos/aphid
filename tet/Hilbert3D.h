/*
 *  Hilbert3D.h
 *  
 *
 *  Created by jian zhang on 6/1/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef TTG_HILBERT_3D_H
#define TTG_HILBERT_3D_H
#include "Scene.h"
#include "QuickSort.h"

namespace ttg {

class Hilbert3D : public Scene {

	int m_N, m_level;
	aphid::Vector3F * m_X;
	aphid::QuickSortPair<int, int> * m_ind;
	
public:
	Hilbert3D();
	virtual ~Hilbert3D();
	
	virtual const char * titleStr() const;
	virtual bool init();
	virtual bool progressForward();
	virtual bool progressBackward();
	virtual void draw(aphid::GeoDrawer * dr);
	
private:
	void generateSamples(int level);
	
		
};

}
#endif