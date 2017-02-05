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

#include <ttg/AdaptiveBccGrid3.h>
#include <ttg/tetrahedron_graph.h>
#include <geom/ConvexShape.h>
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

public:
    TetraMeshBuilder();
    virtual ~TetraMeshBuilder();
    
    template<typename T>
    void buildMesh(T * grid, AdaptiveBccGrid3 * bccg)
    {
        std::vector<ITetrahedron *> tets;
        
        bccg->begin();
        while(!bccg->end() ) {
/// undivided
            if(!bccg->value()->hasChild() ) {
                bccg->value()->connectTetrahedrons(tets, bccg->key(), bccg);
            }
                
            bccg->next();
        }
        
        std::map<int, Vector3F > vertmap;
        bccg->begin();
        while(!bccg->end() ) {
            
            mapNodeInCell(vertmap, bccg->value() );
            
            bccg->next();
        }
        
        const int np = vertmap.size();
        const int nc = tets.size();
        
        grid->create(np, nc);
        
        std::map<int, Vector3F >::iterator it = vertmap.begin();
        for(int i=0;it!=vertmap.end();++it,++i) {
            grid->setPos(it->second, i);
        }
        
        std::vector<ITetrahedron *>::const_iterator itc = tets.begin();
        for(int i=0;itc!=tets.end();++itc,++i) {
            const ITetrahedron * c = *itc;
            grid->setCell(c->iv0, c->iv1, c->iv2, c->iv3, i);
        }
        
        vertmap.clear();
        clearTetra(tets);
        
    }
                
protected:

private:
    void mapNodeInCell(std::map<int, Vector3F > & dst,
                        BccCell3 * cell );
    void clearTetra(std::vector<ITetrahedron *> & tets);
    
};

}
}
#endif
