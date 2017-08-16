/*
 *  SynthesisGroup.h
 *  
 *  instance geom_ind and tm_mat 
 *
 *  Created by jian zhang on 8/16/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_SYNTHESIS_GROUP_H
#define GAR_SYNTHESIS_GROUP_H

#include <vector>

namespace aphid {
class Matrix44F;
}

namespace gar {

class SynthesisGroup {

	std::vector<int> m_geoms;
	std::vector<aphid::Matrix44F > m_tms;
	int m_numInstances;
	
public:
	SynthesisGroup();
	~SynthesisGroup();
	
	const int& numInstances() const;
	
	void addInstance(const int& geom, const aphid::Matrix44F& tm);
/// i-th instance
	void getInstance(int& geom, aphid::Matrix44F& tm, const int& i);
	
};

}

#endif