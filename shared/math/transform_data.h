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

/// x is 3-by-n 3-dimensional points stored columnwise
template<typename T>
inline void transform_data(DenseMatrix<T> & X,
						const Vector3F & sca,
						const Matrix33F & rot,
						const Vector3F & pos,
						const T & noise)
{
	for(int i=0;i<X.numCols();++i) {
		T * c = X.column(i);
		c[0] += GenerateGaussianNoise((T)0.0, noise) ;
		c[1] += GenerateGaussianNoise((T)0.0, noise) ;
		c[2] += GenerateGaussianNoise((T)0.0, noise) ;
		
		Vector3F tv = Vector3F(c[0], c[1], c[2]) * sca;
		tv = rot.transform(tv);
		tv += pos;
		c[0] = tv.x;
		c[1] = tv.y;
		c[2] = tv.z;
	}
}

}
#endif