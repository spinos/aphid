/*
 *  dwt2.cpp
 *  
 *
 *  Created by jian zhang on 9/15/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "dwt2.h"
#include <AllMath.h>

namespace aphid {

namespace wla {

void circleShift1(float * x, const int & n, const int & p)
{
	if(p==0) 
		return;
		
	const int q = Absolute<int>(p);
	float * b = new float[q];
	
	int i;
	if(p<0) {
/// b at beginning
		for(i=0; i<q;++i)
			b[i] = x[i];
			
		for(i=0; i<n-q;++i)
			x[i] = x[i+q];
			
		for(i=0; i<q;++i)
			x[n-q+i] = b[i];
	}
	else {
/// b at end
		for(i=0; i<q;++i)
			b[i] = x[n-q+i];
			
		for(i=n-1; i>=q;--i)
			x[i] = x[i-q];
			
		for(i=0; i<q;++i)
			x[i] = b[i];
	}
	
	delete[] b;
}

void circleShift2(Array2<float> * x, const int & p)
{
	const int & m = x->numRows();
	const int & n = x->numCols();
	
	for(int i=0; i<n; ++i) {
		circleShift1(x->column(i), m, p);
	}
}

}

}