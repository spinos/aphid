/*
 *  GroupSelect.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 1/20/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "GroupSelect.h"
#include <math/PseudoNoise.h>
#include <iostream>

namespace aphid {

GroupSelect::GroupSelect()
{}

GroupSelect::~GroupSelect()
{}

void GroupSelect::createEntityKeys(int n)
{ 
	m_randGroup.reset(new int[n]); 
	PseudoNoise pnoise;
	for(int i = 0;i<n;++i) {
		m_randGroup[i] = pnoise.rint1(i + 2397 * i, n * 5);
	}
}

void GroupSelect::clearGroups()
{ m_groups.clear(); }

void GroupSelect::addGroup(int c)
{ m_groups.push_back(Int2(c, 0) ); }

void GroupSelect::finishGroups()
{
	int b = 0;
	for(int i=0;i<m_groups.size();++i) {
		m_groups[i].y = b;
		b += m_groups[i].x;
	}
	
	const int ng = m_groups.size();
	std::cout<<"\n GroupSelect finish n group "<<ng;
	for(int i=0;i<ng;++i) {
		std::cout<<"\n "<<i<<": "<<m_groups[i].x<<", "<<m_groups[i].y;
	}
}

int GroupSelect::selectInstance(int iGroup, int k) const
{
	if(iGroup >= m_groups.size() ) {
		return 0;
	}
	
	const Int2 & g = m_groups[iGroup];
	return g.y + (k % g.x); 
}

const int & GroupSelect::entityKey(int i) const
{ return m_randGroup[i]; }

}