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

	boost::scoped_ptr<CUDABuffer> m_branchPool;
	boost::scoped_ptr<CUDABuffer> m_leafPool;
	boost::scoped_ptr<CUDABuffer> m_primIndirecion;
	boost::scoped_ptr<CUDABuffer> m_ropes;
	
public:
	CudaNTree(const std::string & name);
	virtual ~CudaNTree();
	
protected:

private:
};

}
