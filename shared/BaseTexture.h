/*
 *  BaseTexture.h
 *  aphid
 *
 *  Created by jian zhang on 1/25/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>
#include <TypedEntity.h>
#include <NamedEntity.h>

namespace aphid {

class BaseTexture : public TypedEntity, public NamedEntity {
public:
	enum TexelFormat {
		FUChar = 1,
		FFloat = 4
	};
	
	enum TexelDepth {
		DOne = 1,
		DThree = 3
	};
	
	BaseTexture();
	virtual ~BaseTexture();
	virtual unsigned numTexels() const;
	virtual void * data();
	virtual void * data() const;
	virtual void sample(const unsigned & faceIdx, float * dst) const;
	virtual void sample(const unsigned & faceIdx, const float & faceU, const float & faceV, float * dst) const;

	void setTextureFormat(TexelFormat f);
	void setTextureDepth(TexelDepth d);
	TexelFormat textureFormat() const;
	TexelDepth textureDepth() const;
	
	void setAllWhite(bool x);
	bool allWhite() const;
	unsigned dataSize() const;
	
	virtual const Type type() const;
protected:

private:
	TexelFormat m_format;
	TexelDepth m_depth;
	bool m_allWhite;
};

}