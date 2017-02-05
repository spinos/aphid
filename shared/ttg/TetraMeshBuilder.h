/*
 *  TetraMeshBuilder.h
 *  
 *
 *  Created by zhang on 17-2-5.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_TTG_TETRA_MESH_BUILDER_H
#define APH_TTG_TETRA_MESH_BUILDER_H

#include <vector>
#include <map>

namespace aphid {

class Vector3F;

namespace cvx {

class Tetrahedron;

}

namespace ttg {

struct ITetrahedron;

class BccCell3;
class AdaptiveBccGrid3;

class TetraMeshBuilder {

    std::vector<ITetrahedron *> m_tets;
    Vector3F * m_pos;
    int m_numVertices;
	
public:
    TetraMeshBuilder();
    virtual ~TetraMeshBuilder();
    
    void buildMesh(AdaptiveBccGrid3 * grid);
    int numTetrahedrons() const;
    const int & numVertices() const;
    const Vector3F * vertices() const;
	const ITetrahedron * tetra(const int & i) const;
    void getTetra(cvx::Tetrahedron & tet,
                const int & i) const;
                
protected:

private:
    void clearTetra();
	void mapNodeInCell(std::map<int, Vector3F > & dst,
                        BccCell3 * cell );
    
};

}
}
#endif
