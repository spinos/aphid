/*
 *  DrawTetrahedron.cpp
 *  
 *
 *  Created by jian zhang on 2/4/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "DrawTetrahedron.h"
#include <geom/ConvexShape.h>
#include <gl_heads.h>

namespace aphid {

DrawTetrahedron::DrawTetrahedron() {}
DrawTetrahedron::~DrawTetrahedron() {}

static const int sTetraLineVertices[12] = {
0, 1,
1, 2,
2, 0,
0, 3,
1, 3,
2, 3
};

void DrawTetrahedron::drawAWireTetrahedron(const cvx::Tetrahedron & tet) const
{
    glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)&tet.X(0) );
    glDrawElements(GL_LINES, 12, GL_UNSIGNED_INT, sTetraLineVertices);
}

static const int sTetraTriangleVertices[12] = {
0, 1, 2,
0, 2, 3,
0, 3, 1,
1, 3, 2
};

void DrawTetrahedron::drawASolidTetrahedron(const cvx::Tetrahedron & tet) const
{
    Vector3F fvpos[12];
    for(int i=0;i<12;++i) {
        fvpos[i] = tet.X(sTetraTriangleVertices[i]);
    }
    
    Vector3F fvnms[12];
    
    Vector3F anm = (tet.X(1) - tet.X(0) ).cross(tet.X(2) - tet.X(0) );
    anm.normalize();
    for(int i=0;i<3;++i) {
        fvnms[i] = anm;
    }
    
    anm = (tet.X(2) - tet.X(0) ).cross(tet.X(3) - tet.X(0) );
    anm.normalize();
    for(int i=3;i<6;++i) {
        fvnms[i] = anm;
    }
    
    anm = (tet.X(3) - tet.X(0) ).cross(tet.X(1) - tet.X(0) );
    anm.normalize();
    for(int i=6;i<9;++i) {
        fvnms[i] = anm;
    }
    
    anm = (tet.X(3) - tet.X(1) ).cross(tet.X(2) - tet.X(1) );
    anm.normalize();
    for(int i=9;i<12;++i) {
        fvnms[i] = anm;
    }
    
    glNormalPointer(GL_FLOAT, 0, (GLfloat*)fvnms );
    glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)fvpos );
    glDrawArrays(GL_TRIANGLES, 0, 12);
}

void DrawTetrahedron::drawAShrinkSolidTetrahedron(const cvx::Tetrahedron & tet,
                            const float & shrink) const
{
    const Vector3F cen = tet.getCenter();
    Vector3F fvpos[12];
    for(int i=0;i<12;++i) {
        fvpos[i] = tet.X(sTetraTriangleVertices[i]);
        fvpos[i] = cen + (fvpos[i] - cen) * shrink;
    }
    
    Vector3F fvnms[12];
    
    Vector3F anm = (tet.X(1) - tet.X(0) ).cross(tet.X(2) - tet.X(0) );
    anm.normalize();
    for(int i=0;i<3;++i) {
        fvnms[i] = anm;
    }
    
    anm = (tet.X(2) - tet.X(0) ).cross(tet.X(3) - tet.X(0) );
    anm.normalize();
    for(int i=3;i<6;++i) {
        fvnms[i] = anm;
    }
    
    anm = (tet.X(3) - tet.X(0) ).cross(tet.X(1) - tet.X(0) );
    anm.normalize();
    for(int i=6;i<9;++i) {
        fvnms[i] = anm;
    }
    
    anm = (tet.X(3) - tet.X(1) ).cross(tet.X(2) - tet.X(1) );
    anm.normalize();
    for(int i=9;i<12;++i) {
        fvnms[i] = anm;
    }
    
    glNormalPointer(GL_FLOAT, 0, (GLfloat*)fvnms );
    glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)fvpos );
    glDrawArrays(GL_TRIANGLES, 0, 12);
}

}