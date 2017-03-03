/*
 *  MeshSampler.h
 *  mama
 *
 *  Created by jian zhang on 12/28/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_MAMA_MESH_SAMPLER_H
#define APH_MAMA_MESH_SAMPLER_H

#include <mama/MeshHelper.h>

namespace aphid {

class MeshSampler : public MeshHelper {

public:
	MeshSampler();

	static void SampleMeshTrianglesInGroup(sdb::VectorArray<cvx::Triangle> & tris,
								BoundingBox & bbox,
							const MDagPath & groupPath);

protected:
	static bool SampleTriangles(sdb::VectorArray<cvx::Triangle> & tris,
						const int & iBegin, const int & iEnd,
						const MObject & shadingEngineNode);
	
};

};

#endif