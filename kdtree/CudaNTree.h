/*
 *  CudaNTree.h
 *  julia
 *
 *  Created by jian zhang on 3/12/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <boost/scoped_ptr.hpp>
#include <HNTree.h>
#include <ConvexShape.h>

namespace aphid {

class CUDABuffer;

class CudaNTree : public HNTree<cvx::Cube, KdNode4 > {

	boost::scoped_ptr<CUDABuffer> m_deviceBranch;
	boost::scoped_ptr<CUDABuffer> m_deviceLeaf;
	boost::scoped_ptr<CUDABuffer> m_deviceIndirection;
	boost::scoped_ptr<CUDABuffer> m_deviceRope;
	
public:
	CudaNTree(const std::string & name);
	virtual ~CudaNTree();
	
	virtual char load();
    
	void * deviceBranch() const;
	void * deviceLeaf() const;
	void * deviceIndirection() const;
	void * deviceRope() const;
	
protected:

private:
};

}
