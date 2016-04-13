/*
 *  CubeRender.h
 *  
 *
 *  Created by jian zhang on 3/2/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <CudaRender.h>
#include <voxTest.h>
#include <boost/scoped_ptr.hpp>

namespace aphid {

class CubeRender : public CudaRender {

	VoxTest m_test;
	boost::scoped_ptr<CUDABuffer> m_devicePyramidPlanes;
	boost::scoped_ptr<CUDABuffer> m_devicePyramidBox;
	
public:
	CubeRender();
	virtual ~CubeRender();
	
	virtual void setBufferSize(const int & w, const int & h);
	virtual void render();
	
};

}