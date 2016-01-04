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
namespace jul {

class QuatJulia {

	Float4 m_c;
	int m_numIter;
	int m_numGrid;
/// un-organized grid
	sdb::HWorldGrid<sdb::HInnerGrid<hdata::TFloat, 3, 256 >, Vector3F > * m_tree;
public:
	QuatJulia(Parameter * param);
	virtual ~QuatJulia();
	
protected:

private:
	void generate();
	float evalAt(const Vector3F & at) const;
};

}