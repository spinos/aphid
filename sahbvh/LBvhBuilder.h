/*
 *  LBvhBuilder.h
 *  testsah
 *
 *  Created by jian zhang on 5/4/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <BvhBuilder.h>
class CUDABuffer;
class LBvhBuilder : public BvhBuilder {
public:
	LBvhBuilder();
	virtual ~LBvhBuilder();
	
	virtual void initOnDevice();
	
	virtual void build(CudaLinearBvh * bvh);
protected:
	
private:
	
private:
	CUDABuffer * m_internalNodeCommonPrefixValues;
	CUDABuffer * m_internalNodeCommonPrefixLengths;
};