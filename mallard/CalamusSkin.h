/*
 *  CalamusSkin.h
 *  mallard
 *
 *  Created by jian zhang on 12/10/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <CollisionRegion.h>
class MlCalamus;
class MlCalamusArray;
class CalamusSkin : public CollisionRegion {
public:
	CalamusSkin();
	virtual ~CalamusSkin();
	
	void getPointOnBody(MlCalamus * c, Vector3F &p) const;
	void getNormalOnBody(MlCalamus * c, Vector3F &p) const;
	
	void tangentSpace(MlCalamus * c, Matrix33F & frm) const;
	void rotationFrame(MlCalamus * c, const Matrix33F & tang, Matrix33F & frm) const;
	void calamusSpace(MlCalamus * c, Matrix33F & frm) const;
	MlCalamusArray * getCalamusArray() const;
	MlCalamus * getCalamus(unsigned idx) const;
	
	void clearFeather();
	void setNumFeathers(unsigned num);
	unsigned numFeathers() const;
	void addFeather(MlCalamus & ori);
	void zeroFeather();
	void reduceFeather(unsigned num);
	
	void clearFaceVicinity();
	void createFaceVicinity();
	void resetFaceVicinity();
	void setFaceVicinity(unsigned idx, float val);
	float faceVicinity(unsigned idx) const;
private:
	MlCalamusArray * m_calamus;
	float * m_perFaceVicinity;
	unsigned m_numFeather;
};