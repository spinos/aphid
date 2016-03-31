/*
 *  JuliaWorld.h
 *  julia
 *
 *  Created by jian zhang on 3/31/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef JULIAWORLD_H
#define JULIAWORLD_H

#include "Parameter.h"

namespace jul {

class JuliaWorld {

public:
    JuliaWorld();
    virtual ~JuliaWorld();
    
	void create(const Parameter & param);
	void insert(const Parameter & param);
	
private:

};

}

#endif