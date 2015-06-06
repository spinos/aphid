/*
 *  HWorld.h
 *  mallard
 *
 *  Created by jian zhang on 9/27/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <HBase.h>

class HWorld : public HBase {
public:
	HWorld();
	virtual char save();
	virtual char load();
	
	std::string modifiedTimeStr() const;
	
private:
	int m_modifiedTime;
};