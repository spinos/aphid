/*
 *  Primitive.h
 *  kdtree
 *
 *  Created by jian zhang on 10/16/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

class Primitive {
public:
	Primitive();
	virtual ~Primitive();
	virtual void name();
	int m_type;
};