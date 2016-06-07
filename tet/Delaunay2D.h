/*
 *  Delaunay2D.h
 *  
 *
 *  Created by jian zhang on 6/1/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef TTG_DELAUNAY_2D_H
#define TTG_DELAUNAY_2D_H
#include "Scene.h"
#include "triangulation.h"
#include "QuickSort.h"

namespace ttg {

class Delaunay2D : public Scene {

	int m_N, m_numTri, m_endTri;
	aphid::Vector3F * m_X;
	aphid::QuickSortPair<int, int> * m_ind;
	ITRIANGLE * m_triangles;
	
public:
	Delaunay2D();
	virtual ~Delaunay2D();
	
	virtual const char * titleStr() const;
	virtual bool init();
	virtual bool progressForward();
	virtual bool progressBackward();
	virtual void draw(aphid::GeoDrawer * dr);
	
private:
	void generateSamples();
	bool triangulate();
	int searchTri(const aphid::Vector3F & p) const;
		
};

}
#endif