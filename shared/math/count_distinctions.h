/*
 *  count_distinctions.h
 *  proxyPaint
 *
 *  number of points far enough from one other
 *  data stored rowwise
 *
 *  Created by jian zhang on 1/22/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_MATH_COUNT_DISTINCTIONS_H
#define APH_MATH_COUNT_DISTINCTIONS_H

#include <math/linearMath.h>
#include <vector>

namespace aphid {

template<typename T>
inline bool far_enough_to_group(const DenseVector<T> & ap,
								const std::vector<DenseVector<T> * > & grps)
{
	T minD = 1e20;
	for(int i=0;i<grps.size();++i) {
		const DenseVector<T> & ag = *grps[i];
		
		T diff = (ap - ag).norm();
		if(minD > diff) {
			minD = diff;
		}
	}
	return minD > 0.4;
}

template<typename T>
inline int count_distinctions(const DenseMatrix<T> & points)
{
	const int d = points.numCols();
	std::vector<DenseVector<T> * > grps;
	
	DenseVector<T> * ag = new DenseVector<T>(d);
	points.extractRowData(ag->raw(), 0);
	grps.push_back(ag);
	
	for(int i=1;i<points.numRows();++i) {
		DenseVector<T> * ig = new DenseVector<T>(d);
		points.extractRowData(ig->raw(), i);
		
		if(far_enough_to_group(*ig, grps) ) {
			grps.push_back(ig);
		} else {
			delete ig;
		}
	}
	
	const int ng = grps.size();
	for(int i=0;i<ng;++i) {
		delete grps[i];
	}
	
	return ng;
}

}

#endif
