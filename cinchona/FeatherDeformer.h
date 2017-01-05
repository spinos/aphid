/*
 *  FeatherDeformer.h
 *  
 *
 *  Created by jian zhang on 1/5/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef FEATHER_DEFORMER_H
#define FEATHER_DEFORMER_H

#include <boost/scoped_array.hpp>

namespace aphid {

class Vector3F;

class Matrix33F;

}

class FeatherMesh;

class FeatherDeformer {

	const FeatherMesh * m_mesh;
	boost::scoped_array<aphid::Vector3F > m_points;
	
public:
	FeatherDeformer(const FeatherMesh * mesh);
	virtual ~FeatherDeformer();
	
	void deform(const aphid::Matrix33F & mat);
	
	const aphid::Vector3F * deformedPoints() const;
	
protected:

private:
};
#endif
