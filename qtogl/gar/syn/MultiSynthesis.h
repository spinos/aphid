/*
 *  MultiSynthesis.h
 *  
 *
 *  Created by jian zhang on 8/16/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_MULTI_SYNTHESIS_H
#define GAR_MULTI_SYNTHESIS_H

#include <vector>

namespace gar {

class SynthesisGroup;

class MultiSynthesis {

	typedef std::vector<SynthesisGroup* > SynthListType;
	SynthListType m_synths;
	
public:
	MultiSynthesis();
	virtual ~MultiSynthesis();

protected:
	SynthesisGroup* addSynthesisGroup();
	const SynthListType& synthsisGroups() const;
	virtual void clearSynths();
	
};

}

#endif
