/*
 *  BarbWorks.h
 *  mallard
 *
 *  Created by jian zhang on 1/1/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <MlCache.h>
#include <LODFn.h>
class MlSkin;
class MlCalamus;
class RenderEngine;
class AdaptableStripeBuffer;
class BlockStripeBuffer;
class BarbWorks : public MlCache, public LODFn {
public:
	BarbWorks();
	virtual ~BarbWorks();
	void setSkin(MlSkin * skin);
	MlSkin * skin() const;
	void createBarbBuffer(RenderEngine * engine);
	void clearBarbBuffer();
	float percentFinished() const;
	unsigned numBlocks() const;
	AdaptableStripeBuffer * block(unsigned idx) const;
	void clearBlock(unsigned idx);
protected:
	unsigned numFeathers() const;
	char isTimeCached(const std::string & stime);
	void openCache(const std::string & stime);
	void closeCache(const std::string & stime);
	void computeFeather(MlCalamus * c);
	void computeFeather(MlCalamus * c, const Vector3F & p, const Matrix33F & space);
private:
	MlSkin * m_skin;
	BlockStripeBuffer * m_stripes;
	float m_percentFinished;
};