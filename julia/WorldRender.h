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
#include <ConvexShape.h>
#include <HWorldGrid.h>
#include <HAssetGrid.h>
#include <WorldManager.h>

namespace aphid {

class WorldRender : public CudaRender {

typedef aphid::sdb::HAssetGrid<aphid::HTriangleAsset, aphid::cvx::Triangle > InnerGridT;
typedef aphid::sdb::HWorldGrid<InnerGridT, aphid::cvx::Triangle > WorldGridT;
typedef aphid::KdNTree<aphid::Voxel, aphid::KdNode4 > TreeT;
	jul::WorldManager<WorldGridT, InnerGridT, TreeT> m_io;
		
public:
	WorldRender(const std::string & filename);
	virtual ~WorldRender();
	
	virtual void setBufferSize(const int & w, const int & h);
	virtual void render();
	
};

}