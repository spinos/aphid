/*
 *  Interpolate1D.h
 *  
 *
 *  Created by jian zhang on 12/17/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_GPR_INTERPOLATE_1D_H
#define APH_GPR_INTERPOLATE_1D_H
#include <math/ATypes.h>
#include <vector>
namespace aphid {

template<typename T>
class DenseVector;

template<typename T>
class DenseMatrix;

namespace gpr {

template<typename T>
class RbfKernel;

template<typename T1, typename T2>
class Covariance;

class Interpolate1D {

	std::vector<Float2 > m_observations;
	Float2 m_bound;
	DenseMatrix<float > * m_xTrain;
	DenseMatrix<float > * m_yTrain;
	RbfKernel<float > * m_rbf;
	Covariance<float, RbfKernel<float > > * m_covTrain;
	DenseVector<float > * m_yMean;
	
public:
	Interpolate1D();
	virtual ~Interpolate1D();
	
	void clearObservations();
	void addObservation(const float & x,
					const float & y);
	void setObservation(const float & x,
					const float & y,
					const int & idx);
	bool learn();
	float predict(const float & x) const;
	int numObservations() const;
	void setBound(const float & lft,
				const float & rgt);
	
protected:

private:

};

}
}
#endif
