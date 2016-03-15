/*
 *  WorldRender.h
 *  julia
 *
 *  Created by jian zhang on 3/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once

#include <CudaRender.h>
#include <NTreeIO.h>
#include <HWorldGrid.h>
#include <HInnerGrid.h>

namespace aphid {

class WorldRender : public CudaRender {

typedef sdb::HWorldGrid<sdb::HInnerGrid<hdata::TFloat, 4, 1024 >, cvx::Sphere > WorldGridT;
typedef HNTree<cvx::Cube, KdNode4 > WorldTreeT;

	NTreeIO m_io;
	sdb::VectorArray<cvx::Cube> m_worldCoord;
	WorldGridT * m_worldGrid;
	WorldTreeT * m_worldTree;
	
public:
	WorldRender(const std::string & filename);
	virtual ~WorldRender();
	
	virtual void setBufferSize(const int & w, const int & h);
	virtual void render();
	
};

}