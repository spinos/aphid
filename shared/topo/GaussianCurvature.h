/*
 *  GaussianCurvature.h
 *  
 *  gaussian curvature K maximum bending (rate of change) of the surface
 *  mean curvature H minimum bending of tangent direction 
 *  using gauss-bonnet scheme
 *  K = (2PI - sigma alpha_i) / (A / 3)
 *  H = (1/4 sigma ||e_i||beta_i) / (A / 3) 
 *  alpha is angle between two successive edges e_i = vvi
 *  beta is angle between normals of two successive neighbor vertices
 *  vi is 1-ring neighbor of v
 *  A is the accumulated areas of triangles around v
 *
 *  reference A comparison of Gaussian and mean curvature estimation methods 
 *  on triangular meshes of range image data
 *
 *  Created by jian zhang on 10/30/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_TOPO_GAUSSIAN_CURVATURE_H
#define APH_TOPO_GAUSSIAN_CURVATURE_H

namespace aphid {

namespace topo {

}

}

#endif
