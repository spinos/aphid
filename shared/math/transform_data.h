/*
 *  transform_data.h
 *  
 *  Created by jian zhang on 12/19/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_TRANSFORM_DATA_H
#define APH_TRANSFORM_DATA_H

#include <math/linearMath.h>
#include <math/Matrix33F.h>
#include <math/miscfuncs.h>

namespace aphid {

template<typename T>
inline void transform_data(DenseMatrix<T> & X,
						const Matrix33F & rot,
						const Vector3F & pos,
						const T & noise)
{
	for(int i=0;i<X.numCols();++i) {
		T * c = X.column(i);
		Vector3F tv(c[0], c[1], c[2]);
		tv = rot.transform(tv);
		tv += pos;
		c[0] = tv.x + GenerateGaussianNoise((T)0.0, noise) ;
		c[1] = tv.y + GenerateGaussianNoise((T)0.0, noise) ;
		c[2] = tv.z + GenerateGaussianNoise((T)0.0, noise) ;
	}
}

}
#endif