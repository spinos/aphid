/*
 *  PatchNeighborRec.h
 *  aphid
 *
 *  Created by jian zhang on 1/25/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <vector>
#include <boost/scoped_array.hpp>

namespace aphid {

class PatchNeighborRec {
public:
	PatchNeighborRec();
	~PatchNeighborRec();
	
	void setEdges(const std::vector<int> & src);
	void setCorners(const std::vector<int> & src, const std::vector<char> & tag);
	
	const unsigned & numEdges() const;
	const unsigned & numCorners() const;
	const int * edges() const;
	const int * corners() const;
	const char * tagCorners() const;
private:
	boost::scoped_array<int> m_edges;
	boost::scoped_array<int> m_corners;
	boost::scoped_array<char> m_tagCorners;
	unsigned m_numEdges, m_numCorners;
};

}