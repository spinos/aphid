#ifndef SIMPLEMESH_H
#define SIMPLEMESH_H

/*
 *  simpleMesh.h
 *  cudabvh
 *
 *  Created by jian zhang on 1/14/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include "BvhTriangleMesh.h"

class SimpleMesh : public BvhTriangleMesh {
public:
	SimpleMesh();
	virtual ~SimpleMesh();
	
	void setAlpha(float x);
	const float alpha() const;
	
	virtual void update();
protected:

private:
	float m_alpha;
};

#endif        //  #ifndef SIMPLEMESH_H

