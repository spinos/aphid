/*
 *  PowellMethod.cpp
 *  aphid
 *
 *  Created by jian zhang on 10/26/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "PowellMethod.h"

PowellMethod::PowellMethod() {}

void PowellMethod::solve(BaseFunction & F, VectorN<double> & x)
{
	const unsigned n = F.ndim();
	unsigned i;
	
	VectorN<double> x0(n);
	x0 = x;
	VectorN<double> * U = new VectorN<double>[n];
	for(i = 0; i < n; i++) {
		U[i].setZero(n);
		*U[i].at(i) = 1.0;
	}
	
	float y = F.f(x);
	float y0 = y;
	float Dx, Dy;
	
	for(int i = 0; i < 12; i++) {
		std::cout<<"cycle "<<i<<" begin\n f["<<x.info()<<"] = "<<F.f(x)<<"\n";
		cycle(F, x, U);
		
		VectorN<double> deltaX = x - x0;
		Dx = deltaX.multiplyTranspose();
		
		if(Dx < 10e-9) break;
		
		y = F.f(x);
		
		Dy = (y - y0) * (y - y0);
		
		if(Dy < 10e-9) break;
		
		y0 = y;
		x0 = x;
		
		std::cout<<"cycle "<<i<<" end\n";
	}
	std::cout<<"powell minimization "<<x.info()<<"\n f["<<x.info()<<"] = "<<F.f(x)<<"\n";
}

void PowellMethod::cycle(BaseFunction & F, VectorN<double> & x, VectorN<double> * U)
{
	const unsigned n = x._ndim;
	VectorN<double> x0(n);
	x0 = x;
	
	VectorN<double> S(n);
	VectorN<double> a(n + 1);

	unsigned i = 0;
	while(i < n) {
		S = U[i];
		*a.at(i) = minimization(F, x, S);
		x = x + S * a[i];
		std::cout<<"Si = "<<S.info();
		std::cout<<" ai = "<<a[i]<<"\n";
		std::cout<<"f["<<x.info()<<"] = "<<F.f(x)<<"\n";
		i++;
	}

	S = x - x0;
	*a.at(n) = minimization(F, x0, S);
	x = x0 + S * a[n];

	std::cout<<"S = "<<S.info();
	std::cout<<" a = "<<a[n]<<"\n";
	std::cout<<"f["<<x.info()<<"] = "<<F.f(x)<<"\n";
	for(i = 1; i < n; i++) U[i-1] = U[i];

	U[n-1] = S;
}

double PowellMethod::minimization(BaseFunction & F, const VectorN<double> & at, const VectorN<double> & along)
{
	m_f = &F;
	m_at = at;
	m_along = along;
	
	return goldenSectionSearch(F.limitLow(), 0, F.limitHigh(), 10e-7);
}

double PowellMethod::taxiCab(double x)
{
	return m_f->f(m_at + m_along * x);
}

double PowellMethod::goldenSectionSearch(double a, double b, double c, double tau) 
{
    double x;
    if (c - b > b - a)
      x = b + ReGoldenRatio * (c - b);
    else
      x = b - ReGoldenRatio * (b - a);
    if (Absolute(c - a) < tau * (Absolute(b) + Absolute(x))) 
      return (c + a) / 2; 

	double tx = taxiCab(x);
	double tb = taxiCab(b);
	if(tx == tb) return x;

    if (tx < tb) {
      if (c - b > b - a) return goldenSectionSearch(b, x, c, tau);
      else return goldenSectionSearch(a, x, b, tau);
    }
    else {
      if (c - b > b - a) return goldenSectionSearch(a, b, x, tau);
      else return goldenSectionSearch(x, b, c, tau);
    }
}