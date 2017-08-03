/*
 *  RodDeformer.h
 *  
 *	storing n state samples
 *
 *  Created by jian zhang on 8/3/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef ROD_DEFORMER_H
#define ROD_DEFORMER_H

#include <pbd/ElasticRodContext.h>
#include "StemDeformer.h"

class RodDeformer : public aphid::pbd::ElasticRodContext, public StemDeformer
{
/// n * (np + ng) points
	boost::scoped_array<float > m_states;
	
public:
	RodDeformer();
	virtual ~RodDeformer();
	
protected:
	void initStates();
/// i-th state
	void saveState(const int& i);
	void loadState(const int& i);
	
private:
};

#endif
