/*
 *  ATriangleMeshGroup.cpp
 *  aphid
 *
 *  Created by jian zhang on 7/6/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "ATriangleMeshGroup.h"
namespace aphid {

ATriangleMeshGroup::ATriangleMeshGroup() {}
ATriangleMeshGroup::~ATriangleMeshGroup() {}
	
void ATriangleMeshGroup::create(unsigned np, unsigned nt, unsigned ns)
{
	ATriangleMesh::create(np, nt);
	AStripedModel::create(ns);
}

}
	