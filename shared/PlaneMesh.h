/*
 *  PlaneMesh.h
 *  aphid
 *
 *  Created by jian zhang on 10/24/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <PatchMesh.h>

class PlaneMesh : public PatchMesh {
public:
	PlaneMesh(const Vector3F & bottomLeft, const Vector3F & bottomRight, const Vector3F & topRight, const Vector3F & topLeft, unsigned gu, unsigned gv);
	virtual ~PlaneMesh();
private:
};