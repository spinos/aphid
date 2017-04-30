/*
 *  MeshShape.h
 *  wheeled
 *
 *  Created by jian zhang on 6/1/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <Mesh.h>
namespace caterpillar {
class MeshShape : public Mesh {
public:
	MeshShape();
	virtual ~MeshShape();
	
protected:
	btBvhTriangleMeshShape* createCollisionShape();
	void setMargin(const float & x);
	const float margin() const;
private:
	btTriangleIndexVertexArray* m_indexVertexArrays;
	float m_margin;
};
}