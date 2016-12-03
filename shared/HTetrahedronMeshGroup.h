#ifndef APH_H_TETRAHEDRON_MESH_GROUP_H
#define APH_H_TETRAHEDRON_MESH_GROUP_H
#include "HTetrahedronMesh.h"

namespace aphid {

class ATetrahedronMeshGroup;
class HTetrahedronMeshGroup : public HTetrahedronMesh {
public:
	HTetrahedronMeshGroup(const std::string & path);
	virtual ~HTetrahedronMeshGroup();
	
	virtual char verifyType();
	virtual char save(ATetrahedronMeshGroup * tetra);
	virtual char load(ATetrahedronMeshGroup * tetra);
};

}
#endif

