/*
 *  D3Q19DF.cpp
 *  
 *
 *  Created by jian zhang on 1/19/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#include "D3Q19DF.h"

namespace aphid {

namespace lbm {

const float D3Q19DF::w_alpha[19] = { (1./3.), /// 0 length
	(1./18.),(1./18.),(1./18.),(1./18.),(1./18.),(1./18.), /// 1 length
	(1./36.),(1./36.),(1./36.),(1./36.),(1./36.),(1./36.), /// squared 2 length
	(1./36.),(1./36.),(1./36.),(1./36.),(1./36.),(1./36.), };
	
const float D3Q19DF::e_alpha[3][19] = {
{ 0,1.f,-1.f,     0, 0,    0, 0, 1.f,-1.f, 1.f,-1.f, 1.f,-1.f, 1.f,-1.f,         0, 0, 0, 0 },
{ 0,    0, 0, 1.f,-1.f,    0, 0, 1.f, 1.f,-1.f,-1.f,         0, 0, 0, 0, 1.f,-1.f, 1.f,-1.f },
{ 0,    0, 0,     0, 0,1.f,-1.f,         0, 0, 0, 0, 1.f, 1.f,-1.f,-1.f, 1.f, 1.f,-1.f,-1.f },
};

const int D3Q19DF::inv_e_alpha[3][19] = {
{ 0,-1, 1, 0, 0, 0, 0,-1, 1,-1, 1,-1, 1,-1, 1, 0, 0, 0, 0 },
{ 0, 0, 0,-1, 1, 0, 0,-1,-1, 1, 1, 0, 0, 0, 0,-1, 1,-1, 1 },
{ 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0,-1,-1, 1, 1,-1,-1, 1, 1 },
};

D3Q19DF::D3Q19DF()
{}

void D3Q19DF::SetWi(float* q, const int& n, const int& i)
{
	for(int j=0;j<n;++j) {
		q[j] = w_alpha[i];
	}
}

void D3Q19DF::GetStreamDirection(int* c, const int& i)
{
	c[0] = inv_e_alpha[0][i];
	c[1] = inv_e_alpha[1][i];
	c[2] = inv_e_alpha[2][i];
}

void D3Q19DF::IncomingBC(float* f_i[], const float* u, const float& rho, const int& ind)
{
	for(int i=1;i< 19;++i) {
		float eu = e_alpha[0][i] * u[0] 
				+ e_alpha[1][i] * u[1] 
				+ e_alpha[2][i] * u[2];
				
		float f_eq = w_alpha[i] * ( rho + 3.f * eu + 3.f * eu * eu );
		
		f_i[i][ind] += f_eq;
	}
}

void D3Q19DF::Relaxing(float* f_i[], const float* u, const float& uu, const float& rho, 
						const float& omega, const int& ind)
{
	for(int i=0;i< 19;++i) {
		float eu = e_alpha[0][i] * u[0] 
				+ e_alpha[1][i] * u[1] 
				+ e_alpha[2][i] * u[2];

/// dx/dt = 1				
		float f_eq = w_alpha[i] * rho * (1.f + 3.f * eu + 4.5f * eu * eu - 1.5f * uu );
/// tau = .66		
		f_i[i][ind] = (1.f - omega) * f_i[i][ind] + omega * f_eq;
	}
}

void D3Q19DF::Density(float& rho, float* f_i[],
						const int& ind)
{
	rho = 0.f;
	for(int i=0;i<19;++i) {
		rho += f_i[i][ind];
	}
}

void D3Q19DF::IncompressibleVelocity(float* u, float& rho, float* f_i[],
						const int& ind)
{
	u[0] = u[1] = u[2] = 0.f;
	rho = 0.f;
	for(int i=0;i<19;++i) {
		const float& fi = f_i[i][ind];
		rho += fi;
		u[0] += e_alpha[0][i] * fi;
		u[1] += e_alpha[1][i] * fi;
		u[2] += e_alpha[2][i] * fi;
	}
}

void D3Q19DF::CompressibleVelocity(float* u, float& rho, float* f_i[],
						const int& ind)
{
	IncompressibleVelocity(u, rho, f_i, ind);
	
	if(rho > 0.f) {
		u[0] /= rho;
		u[1] /= rho;
		u[2] /= rho;
	}
}

void D3Q19DF::DiscretizeVelocity(float* f_i[], const float& rho, const float* u,
				const int& ind)
{
	for(int i=0;i<19;++i) {
		float eu = e_alpha[0][i] * u[0] 
				+ e_alpha[1][i] * u[1] 
				+ e_alpha[2][i] * u[2];
/// rho <- 1				
		f_i[i][ind] = w_alpha[i] * rho * (1.f +  3.f * eu);
	}
}

void D3Q19DF::Equilibrium(float* f_i[], const float& rho, const float* u, const float& uu,
						const int& ind)
{
	for(int i=0;i<19;++i) {
		float eu = e_alpha[0][i] * u[0] 
				+ e_alpha[1][i] * u[1] 
				+ e_alpha[2][i] * u[2];
				
		f_i[i][ind] = w_alpha[i] * rho *(1.f + 3.f * eu + 4.5f * eu * eu - 1.5f * uu );
	}
}

}

}