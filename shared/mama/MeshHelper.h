/*
 *  MeshHelper.h
 *  proxyPaint
 *
 *  Created by jian zhang on 12/28/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_MAMA_MESH_HELPER_H
#define APH_MAMA_MESH_HELPER_H

class MDagPath;
class MIntArray;
class MVectorArray;
class MObject;

namespace aphid {

class ATriangleMesh;

class MeshHelper {

public:
	MeshHelper();
	
	static unsigned GetMeshNv(const MObject & meshNode);
	static void CountMeshNv(int & nv,
					const MDagPath & meshPath);
	static void GetMeshTriangles(MIntArray & triangleVertices,
							const MDagPath & meshPath); 
	static void GetMeshTrianglesInGroup(MIntArray & triangleVertices,
							const MDagPath & groupPath);
/// pnts is 3-by-np matrix, points stored columnwise
/// scatter to n_tri_vert pos
	static void ScatterTriangleVerticesPosition(MVectorArray & pos,
						const float * pnts, const int & np,
						const MIntArray & triangleVertices,
						const int & nind);
	static void CalculateTriangleVerticesNormal(MVectorArray & nms,
						const float * pnts, const int & np,
						const MIntArray & triangleVertices,
						const int & nind);
	static void UpdateMeshTriangleUVs(ATriangleMesh * trimesh,
						const MObject & meshNode);

};

};

#endif