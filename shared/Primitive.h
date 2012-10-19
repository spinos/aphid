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

	void setType(short t);
	const short & getType() const;
	
	void setGeom(char * data);
	char *geom();
private:
	unsigned m_type;
	char *m_geom;
};