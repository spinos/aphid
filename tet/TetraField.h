/*
 *  TetraField.h
 *  foo
 *
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include "GridMaker.h"
#include <ADistanceField.h>
#include <ConvexShape.h>

namespace ttg {

class TetraField : public GridMaker, public aphid::ADistanceField {

public:
	TetraField();
	virtual ~TetraField();
	
/// push tetra node and edge to graph
	void buildGraph();
	
	template<typename Tf>
	void calculateDistance(const Tf * func)
	{
		typename aphid::cvx::Tetrahedron;
		aphid::cvx::Tetrahedron tetshp;
/// intersect any tetra
		const int nt = numTetrahedrons();
		int i = 0;
		for(;i<nt;++i) {
			
			getTetraShape(tetshp, i);
			
			if(func-> template intersect <aphid::cvx::Tetrahedron>(&tetshp) ) {
				markTetraOnFront(i);
			}
		}
		
		calculateDistanceNodeOnFront(func);
	}
		
	void verbose();
	
protected:
	void getTetraShape(aphid::cvx::Tetrahedron & b, const int & i) const;
	void markTetraOnFront(const int & i);
	
private:
	void pushIndices(const std::vector<int> & a,
					std::vector<int> & b);
	
};

}