/*
 *  dtdwt2.h
 *  
 *  2-D dual tree discrete wavelet transform
 *  Created by jian zhang on 9/16/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef WLA_DT_DWT_2D_H
#define WLA_DT_DWT_2D_H

#include "fltbanks.h"

namespace aphid {

namespace wla {

class DualTree2 {

#define DT2_MAX_N_STAGE 6
/// lohi hilo hihi for up [0:2] and down [3:5] tree
	Array3<float> m_w[DT2_MAX_N_STAGE][6];
	int m_lastStage, m_numRanks;

public:
	DualTree2();
	virtual ~DualTree2();
	
/// x input signal
/// nstage number of stages, truncate to DT2_MAX_N_STAGE
	void analize(const Array3<float> & x, const int & nstage);
/// y output signal
	void synthesize(Array3<float> & y);
	
	const int & lastStage() const;
/// i stage
/// j [0,1] up or down
/// k [0,2] lohi hilo hihi
	const Array3<float> & stageBand(const int & i, const int & j, const int & k ) const;
	const Array3<float> & lastStageBand(const int & j) const;

	void scaleUp(int level, float scaling);
	
/// nearest neightbor search
/// a 16x16 patch will shrink to 1x1 after 4 stages
/// provide the coordinate in level 4 texture
/// Ts input tree
/// Crow input row coordinate of texel in last stage of Ts
/// Ccol input column coordinate of texel in last stage of Ts
	void nns(const DualTree2 & ts,
			const int & crow, const int & ccol) const;
	
protected:

private:
	void createStage(int j, int m, int n, int p);
	
};

}

}
#endif