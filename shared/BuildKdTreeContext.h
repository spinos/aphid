/*
 *  BuildKdTreeContext.h
 *  kdtree
 *
 *  Created by jian zhang on 10/20/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <BuildKdTreeStream.h>
#include <PrimBoundary.h>

namespace aphid {

class BuildKdTreeContext : public PrimBoundary {

public:
	BuildKdTreeContext();
	BuildKdTreeContext(BuildKdTreeStream &data, const BoundingBox & b);
	~BuildKdTreeContext();
	
	bool decompressPrimitives(bool force=false);
	
	static BuildKdTreeContext * GlobalContext;
	
private:
	
private:
	
};

}