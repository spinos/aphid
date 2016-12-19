/*
 *  generate_data.h
 *  
 *	n point data set
 *
 *  Created by jian zhang on 12/19/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_MATH_GENERATE_DATA_H
#define APH_MATH_GENERATE_DATA_H

#include <math/linearMath.h>
#include <math/miscfuncs.h>
#include <string>

namespace aphid {
   
template<typename T>
inline void swiss_roll(DenseMatrix<T> & X, const int & n,
						const T & noise)
{
	T t;
	T height;
	X.resize(3, n);
	for(int i=0;i<n;++i) {
		t = PI * RandomF01() * 4.0;
		height = GenerateGaussianNoise((T)2.0, (T)1.0);
		T * e = X.column(i);
		e[0] = 0.5 * t * cos(t) + GenerateGaussianNoise((T)0.0, noise); 
		e[1] = height + GenerateGaussianNoise((T)0.0, noise);
		e[2] = 0.5 * t * sin(t) + GenerateGaussianNoise((T)0.0, noise);
	}
}

template<typename T>
inline void helix(DenseMatrix<T> & X, const int & n,
						const T & noise)
{
	T t;
	X.resize(3, n);
	for(int i=0;i<n;++i) {
		t = (T)(i+1)/(T)n * PI;
		T * e = X.column(i);
		e[0] = ((T)2.0 + cos(t * (T)8.0) ) * cos(t) + GenerateGaussianNoise((T)0.0, noise); 
		e[1] = ((T)2.0 + cos(t * (T)8.0) ) * sin(t) + GenerateGaussianNoise((T)0.0, noise);
		e[2] = sin(t * (T)8.0) + GenerateGaussianNoise((T)0.0, noise);
	}
}

template<typename T>
inline void generate_data(const std::string & name,
						DenseMatrix<T> & X, const int & n,
						const T & noise)
{
	if(name == "swiss_roll") {
		swiss_roll(X, n, noise);
	} else {
		helix(X, n, noise);
	}
}

}
#endif
