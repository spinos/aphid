#ifndef ATETRAHEDRONMESHGROUP_H
#define ATETRAHEDRONMESHGROUP_H

#include <geom/ATetrahedronMesh.h>
#include <foundation/AStripedModel.h>

namespace aphid {

class ATetrahedronMeshGroup : public ATetrahedronMesh, public AStripedModel {
public:
	ATetrahedronMeshGroup();
	virtual ~ATetrahedronMeshGroup();
	
	void create(unsigned np, unsigned nt, unsigned ns);
	
    virtual std::string verbosestr() const;
private:

};

}
#endif        //  #ifndef ATETRAHEDRONMESHGROUP_H

