/*
 *  D3Q19DF.h
 *  
 *
 *  Created by jian zhang on 1/19/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_LBM_D3Q19_DF_H
#define APH_LBM_D3Q19_DF_H

namespace aphid {

namespace lbm {

class D3Q19DF {

public:
	D3Q19DF();
	
/// q <- w_i n elements
	static void SetWi(float* q, const int& n, const int& i);
/// inverse of c_i	
	static void GetStreamDirection(int* c, const int& i);
	
	static void IncomingBC(float* f_i[], const float* u, const float& rho,
						const int& ind);
/// relax f_i towards the equilibrium value	
/// f_i^eq <- w_i rho (1 - 3 / 2 u.u + 3 e_i.u + 9 / 2 (e_i.u)^2)
	static void Relaxing(float* f_i[], const float* u, const float& uu, const float& rho, 
						const int& ind);
						
	static void IncompressibleVelocity(float* u, float& rho, float* f_i[],
						const int& ind);
/// density rho <- sigma (f_i)	
/// velocity u <- 1 / rho sigma (f_i c_i)
/// at site ind						
	static void CompressibleVelocity(float* u, float& rho, float* f_i[],
						const int& ind);
						
	static void DiscretizeVelocity(float* f_i[], const float* u,
						const int& ind);

/// the equilibrium distribution f_i at zero velocity w_i
	static const float w_alpha[19];
/// the discrete set of microscopic velocities e or c_i
	static const float e_alpha[3][19];
/// inverse direction to stream
	static const int inv_e_alpha[3][19];
	
};

}

}

#endif