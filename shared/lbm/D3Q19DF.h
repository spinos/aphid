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
/// f_i(x, t+1) <- ( 1 - omega )  f_i(x, t) + omega f_i^eq (rho, u)
/// omega <- 1 / tau, range [0, 2] 0 is high viscosity while values near 2 result in more turbulent flows
/// instable when close to 2
/// 0.7 > tau > 0.5 is required for the simualtion to remain stable
	static void Relaxing(float* f_i[], const float* u, const float& uu, const float& rho, 
						const float& omega, const int& ind);
/// valid only in the incompressible limit
/// convergence of the solution to the incompressible Navier-Stokes equations
/// for the Mach number become small enough to remove compressibility effects
/// momentum density rho u <- sigma (f_i c_i)
/// Courant-Friedrich-Lewy (CFL) number (proportional to dx/dt) always equals 1	
/// unstable when dx/dt close to 0.3 lu, omega = 1.9					
	static void IncompressibleVelocity(float* u, float& rho, float* f_i[],
						const int& ind);
/// density rho <- sigma (f_i)	
	static void Density(float& rho, float* f_i[],
						const int& ind);
/// velocity u <- 1 / rho sigma (f_i c_i)
/// at site ind						
	static void CompressibleVelocity(float* u, float& rho, float* f_i[],
						const int& ind);
						
	static void DiscretizeVelocity(float* f_i[], const float& rho, const float* u,
						const int& ind);
/// f_i^eq by rho and u at ind-th site
/// f_i^eq <- w_i rho (1 - 3 / 2 u.u + 3 e_i.u + 9 / 2 (e_i.u)^2)
	static void Equilibrium(float* f_i[], const float& rho, const float* u, const float& uu,
						const int& ind);
/// collision ind from begin to end-1						
	static void ComputeCollision(float* f_i[], const float& omega, const int& begin, const int& end);
/// streaming by direction c_i in z from begin to end-1
	static void ComputeStreaming(float* f_i, const float* tmp, const int* c_i, 
						const int& zbegin, const int& zend, const int* dim);
/// momentum exchange between fluid and particle

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