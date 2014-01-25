/*
 *  PatchTexture.h
 *  aphid
 *
 *  Created by jian zhang on 1/25/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>
#include <boost/scoped_array.hpp>
class PatchMesh;
class PatchTexture {
public:
	PatchTexture();
	virtual ~PatchTexture();
	void create(PatchMesh * mesh);
	
	const unsigned & numColors() const;
	Float3 * colors();
	Float3 * patchColor(const unsigned & idx);
	void sample(const unsigned & faceIdx, const float & faceU, const float & faceV, Float3 & dst) const;
protected:

private:
	boost::scoped_array<Float3> m_colors;
	unsigned m_numColors;
};