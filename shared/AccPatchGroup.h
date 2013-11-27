/*
 *  AccPatchGroup.h
 *  aphid
 *
 *  Created by jian zhang on 11/28/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once

#include <AllMath.h>
class AccPatch;

class AccPatchGroup {
public:
	AccPatchGroup();
	virtual ~AccPatchGroup();
	void createAccPatches(unsigned n);
	AccPatch* beziers() const;
	
private:
	AccPatch* m_bezier;
};