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
#include <BaseTexture.h>
#include <boost/scoped_array.hpp>
class PatchMesh;
class PatchTexture : public BaseTexture {
public:
	PatchTexture();
	virtual ~PatchTexture();
	void create(PatchMesh * mesh);
	
	int resolution() const;
	
	virtual unsigned numTexels() const;
	virtual void * data();
	virtual void * data() const;
	Float3 * patchColor(const unsigned & idx);
	void fillPatchColor(const unsigned & faceIdx, const Float3 & fillColor);
	
	virtual void sample(const unsigned & faceIdx, float * dst) const;
	virtual void sample(const unsigned & faceIdx, const float & faceU, const float & faceV, float * dst) const;
protected:

private:
	void toUChar(const Float3 & src, unsigned char * dst) const;
	void toFloat(unsigned char * src, Float3 & dst) const;
	boost::scoped_array<Float3> m_colors;
	unsigned m_numColors;
};