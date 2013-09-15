/*
 *  MlSkin.h
 *  mallard
 *
 *  Created by jian zhang on 9/14/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include "MlCalamusArray.h"

class AccPatchMesh;
class MlSkin {
public:
	MlSkin();
	virtual ~MlSkin();
	void setBodyMesh(AccPatchMesh * mesh);
	void addCalamus(MlCalamus & ori);
	
	unsigned numFeathers() const;
	MlCalamus * getCalamus(unsigned idx) const;
	
	AccPatchMesh * bodyMesh() const;
	
	void verbose() const;
private:
	MlCalamusArray m_calamus;
	unsigned m_numFeather;
	AccPatchMesh * m_body;
	unsigned * m_faceCalamusStart;
};