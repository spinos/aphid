/*
 *  StemDeformer.h
 *  
 *  bind one point to a segment on line, no blending 
 *
 *  Created by jian zhang on 8/3/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef STEM_DEFORMER_H
#define STEM_DEFORMER_H

#include <math/Matrix44F.h>
#include <boost/scoped_array.hpp>

class StemDeformer 
{
/// in bind space
	boost::scoped_array<aphid::Vector3F > m_localPnt;
/// deformed result
	boost::scoped_array<aphid::Vector3F > m_dfmdPnt;
/// tm ind per pnt
	boost::scoped_array<int > m_bindInd;
	boost::scoped_array<aphid::Matrix44F > m_tm;
	int m_numSpaces;
	int m_numPoints;
	
public:
	StemDeformer();
	virtual ~StemDeformer();

/// num_points num_spaces	
	void createDeformer(const int& np, const int& ns);
	virtual bool solve();
	
	const int& numPoints() const;
	const int& numSpaces() const;
	
	const aphid::Vector3F* deformedPnt() const;
	
protected:
	int* bindInds();
	aphid::Matrix44F* spaces();
	
private:
};

#endif
