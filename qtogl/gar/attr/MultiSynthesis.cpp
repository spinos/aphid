/*
 *  MultiSynthesis.cpp
 *  
 *
 *  Created by jian zhang on 8/16/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "MultiSynthesis.h"
#include "SynthesisGroup.h"

namespace gar {

MultiSynthesis::MultiSynthesis()
{}

MultiSynthesis::~MultiSynthesis()
{
	clearSynths();
}

SynthesisGroup* MultiSynthesis::addSynthesisGroup()
{
	SynthesisGroup* g = new SynthesisGroup;
	m_synths.push_back(g);
	return g;
}

void MultiSynthesis::clearSynths()
{
	SynthListType::iterator it = m_synths.begin();
	for(;it!=m_synths.end();++it) {
		delete *it;
	}
	m_synths.clear();
	
}

const MultiSynthesis::SynthListType& MultiSynthesis::synthsisGroups() const
{ return m_synths; }

}