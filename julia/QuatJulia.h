#ifndef QUATJULIA_H
#define QUATJULIA_H

/*
 *  QuatJulia.h
 *  
 *
 *  Created by jian zhang on 1/1/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include <AllMath.h>
#include "Parameter.h"
#include <HInnerGrid.h>
#include <HWorldGrid.h>
#include <ConvexShape.h>

namespace jul {

using namespace aphid;

class QuatJulia {

	Float4 m_c;
	int m_numIter;
	int m_numGrid;
	float m_scaling;
/// un-organized grid
	sdb::HWorldGrid<sdb::HInnerGrid<hdata::TFloat, 4, 256 >, cvx::Sphere > * m_tree;
public:
	QuatJulia(Parameter * param);
	virtual ~QuatJulia();
	
protected:

private:
	void generate();
	float evalAt(const Vector3F & at) const;
};

}
#endif        //  #ifndef QUATJULIA_H
