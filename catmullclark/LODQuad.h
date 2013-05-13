#ifndef LODQUAD_H
#define LODQUAD_H

/*
 *  LODQuad.h
 *  easymodel
 *
 *  Created by jian zhang on 11/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
class Vector3F;
class LODQuad {
public:
	LODQuad();
	virtual ~LODQuad();
	
	void setCorner(Vector3F p, int i);
	Vector3F getCorner(int i) const;
	
	void setDetail(float d, int i);
	void setUniformDetail(float d);
	float getDetail(int i) const;
	
	void evaluateSurfaceLOD(float u, float v, float * detail) const;
	float getMaxLOD() const;
	float getMaxEdgeLength() const;
	
	Vector3F _corners[4];
	float _details[4];
};
#endif        //  #ifndef LODQUAD_H
