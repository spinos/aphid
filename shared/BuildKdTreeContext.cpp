/*
 *  BuildKdTreeContext.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/20/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "BuildKdTreeContext.h"
#include "Geometry.h"
#include <VectorArray.h>

namespace aphid {

BuildKdTreeContext * BuildKdTreeContext::GlobalContext = NULL;

BuildKdTreeContext::BuildKdTreeContext() 
{}

BuildKdTreeContext::BuildKdTreeContext(BuildKdTreeStream &data, const BoundingBox & b)
{
	setBBox(b);
	
	clearPrimitive();
	
	int igeom, icomp;
	const sdb::VectorArray<Primitive> &primitives = data.primitives();
	
	const unsigned n = data.getNumPrimitives();
	for(unsigned i=0;i<n; i++) {
		addPrimitive(i);
		
		Primitive *p = primitives.get(i);
		
		p->getGeometryComponent(igeom, icomp);

		BoundingBox ab = data.calculateComponentBox(igeom, icomp);
		
		ab.expand(1e-6f);
		
		addPrimitiveBox(ab );
	}
	
	if(numPrims() > 1024) compressPrimitives();
	
}

BuildKdTreeContext::~BuildKdTreeContext() 
{}

bool BuildKdTreeContext::decompressPrimitives(bool force)
{ 
	if(!force) countPrimsInGrid();
	return decompress(GlobalContext->primitiveBoxes(), force);
}

}