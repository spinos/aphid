/*
 *  PatchNeighborRec.cpp
 *  aphid
 *
 *  Created by jian zhang on 1/25/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "PatchNeighborRec.h"

namespace aphid {

PatchNeighborRec::PatchNeighborRec() {}
PatchNeighborRec::~PatchNeighborRec() {}

void PatchNeighborRec::setEdges(const std::vector<int> & src)
{
	m_numEdges = src.size();
	m_edges.reset(new int[m_numEdges]);
	std::vector<int>::const_iterator it = src.begin();
	int i = 0;
	for(; it != src.end(); ++it, ++i) m_edges.get()[i] = *it;
}

void PatchNeighborRec::setCorners(const std::vector<int> & src, const std::vector<char> & tag)
{
	m_numCorners = src.size();
	m_corners.reset(new int[m_numCorners]);
	std::vector<int>::const_iterator it = src.begin();
	int i = 0;
	for(; it != src.end(); ++it, ++i) m_corners.get()[i] = *it;
	
	m_tagCorners.reset(new char[m_numCorners]);
	
	std::vector<char>::const_iterator itt = tag.begin();
	i = 0;
	for(; itt != tag.end(); ++itt, ++i) m_tagCorners.get()[i] = *itt;
}

const unsigned & PatchNeighborRec::numEdges() const { return m_numEdges; }
const unsigned & PatchNeighborRec::numCorners() const { return m_numCorners; }

const int * PatchNeighborRec::edges() const { return m_edges.get(); }
const int * PatchNeighborRec::corners() const { return m_corners.get(); }
const char * PatchNeighborRec::tagCorners() const { return m_tagCorners.get(); }

}