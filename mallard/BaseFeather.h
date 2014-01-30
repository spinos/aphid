/*
 *  BaseFeather.h
 *  mallard
 *
 *  Created by jian zhang on 12/21/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>
#include <BoundingRectangle.h>

class BaseFeather {
public:
	BaseFeather();
	virtual ~BaseFeather();
	
	virtual void changeNumSegment(int d);
	virtual void createNumSegment(short x);
	virtual void computeLength();
	virtual void computeTexcoord();
	virtual void translateUV(const Vector2F & d);
	
	short numSegment() const;
	BoundingRectangle getBoundingRectangle() const;
	
	float shaftLength() const;
	float featherWidth() const;
	
	float * quilly();
    float * getQuilly() const;
	
	Vector2F * uvDisplace();
    Vector2F * uvDisplaceAt(short seg, short side);
    Vector2F * getUvDisplaceAt(short seg, short side) const;
	
	Vector2F baseUV() const;
	void setBaseUV(const Vector2F & d);
	
	Vector2F * segmentQuillTexcoord(short seg);
	Vector2F * segmentVaneTexcoord(short seg, short side, short idx);

	void computeBounding();
	
	unsigned numVaneVertices() const;
	unsigned numWorldP() const;

	float* selectVertexInUV(const Vector2F & p, bool & yOnly, Vector2F & wp);
	Vector3F * normal(unsigned seg);
	void setType(short x);
	short type() const;
protected:	
	virtual void simpleCreate(int ns);

private:
	Vector2F * texcoord();
	void presetQuill();
	void presetBarb();
private:
	BoundingRectangle m_brect;
	Vector2F m_uv;
	Vector2F * m_uvDisplace;
	float *m_quilly;
	Vector2F * m_st;
	Vector3F * m_segementNormals;
	float m_shaftLength;
	short m_numSeg, m_type;
};