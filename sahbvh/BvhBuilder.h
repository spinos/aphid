/*
 *  BvhBuilder.h
 *  testsah
 *
 *  Created by jian zhang on 5/4/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
class CudaReduction;
class CudaLinearBvh;
class BvhBuilder {
public:
	BvhBuilder();
	virtual ~BvhBuilder();
	
	virtual void initOnDevice();
	
	virtual void build(CudaLinearBvh * bvh);
protected:
	CudaReduction * reducer();
private:
	
private:
	CudaReduction * m_findMaxDistance;
};