/*
 *  Delaunay3D.h
 *  
 *
 *  Created by jian zhang on 6/1/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef TTG_DELAUNAY_3D_H
#define TTG_DELAUNAY_3D_H
#include "Scene.h"
#include "tetrahedralization.h"
#include "QuickSort.h"

namespace ttg {

class Delaunay3D : public Scene {

	int m_N, m_numTet, m_endTet;
	aphid::Vector3F * m_X;
	aphid::QuickSortPair<int, int> * m_ind;
	ITetrahedron * m_tets;
	
public:
	Delaunay3D();
	virtual ~Delaunay3D();
	
	virtual const char * titleStr() const;
	virtual bool init();
	virtual bool progressForward();
	virtual bool progressBackward();
	virtual void draw(aphid::GeoDrawer * dr);
	
private:
	void generateSamples();
	bool tetrahedralize();
	int searchTet(const aphid::Vector3F & p) const;
		
};

}
#endif