/*
 *  dtdwt1.h
 *  
 *  1-D dual tree discrete wavelet transform
 *  Created by jian zhang on 9/16/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef WLA_DT_DWT_1D_H
#define WLA_DT_DWT_1D_H

#include "fltbanks.h"

namespace aphid {

namespace wla {

class DualTree {

#define DT_MAX_N_STAGE 6
	VectorN<float> m_w[DT_MAX_N_STAGE][2];
	int m_numStages;

public:
	DualTree();
	virtual ~DualTree();
	
/// x input signal
/// nstage number of stages, truncate to DT_MAX_N_STAGE
	void analize(const VectorN<float> & x, const int & nstage);
	void synthesize(VectorN<float> & y);
	
	const int & numStages() const;
/// i stage
/// j [0,1] up or down
	const VectorN<float> & stage(const int & i, const int & j ) const;
	
protected:

private:

};

}

}
#endif