/*
 *  PowellMethod.h
 *  aphid
 *
 *  Created by jian zhang on 10/26/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <BaseFunction.h>

class PowellMethod {
public:
	PowellMethod();
	
	void solve(BaseFunction & F, Vector2F & x);
private:
	void cycle(BaseFunction & F, Vector2F & x, Vector2F & S0, Vector2F & S1);
};