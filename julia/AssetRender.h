/*
 *  AssetRender.h
 *  
 *
 *  Created by jian zhang on 3/2/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef JUL_ASSET_RENDER_H
#define JUL_ASSET_RENDER_H
#include <CudaRender.h>
#include <boost/scoped_ptr.hpp>
#include <Container.h>
#include <ConvexShape.h>
#include <CudaNTree.h>

namespace aphid {

class AssetRender : public CudaRender {

	Container<cvx::Triangle > m_container;

typedef CudaNTree<Voxel, KdNode4> CuTreeT;
	boost::scoped_ptr<CuTreeT> m_cuTree;
	
public:
	AssetRender();
	virtual ~AssetRender();
	
	bool load(const std::string & filename, const int & level);
	
	virtual void setBufferSize(const int & w, const int & h);
	virtual void render();
/// override baseview
	virtual void frameAll();
	
};

}
#endif