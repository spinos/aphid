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

void PowellMethod::solve(BaseFunction & F, Vector2F & x)
{
	//x.set(2.55, 1.75);
	x.set(.5, .5);
	//x.set(.99, 1.01);
	
	Vector2F x0 = x;
	Vector2F S0(1, 0), S1(0, 1);
	float y = F.f(x);
	float y0 = y;
	float Dx, Dy;
	for(int i = 0; i < 12; i++) {
		std::cout<<"cycle "<<i<<"\n";
		cycle(F, x, S0, S1);
		
		Dx = (x.x - x0.x) * (x.x - x0.x) + (x.y - x0.y) * (x.y - x0.y);
		
		if(Dx < 10e-9) break;
		
		y = F.f(x);
		
		Dy = (y - y0) * (y - y0);
		
		if(Dy < 10e-9) break;
		
		y0 = y;
		x0 = x;
	}
	x.verbose("minimization");
}

void PowellMethod::cycle(BaseFunction & F, Vector2F & x, Vector2F & S0, Vector2F & S1)
{
	const Vector2F x0 = x;
	std::cout<<"cycle begin\n f[("<<x.x<<" , "<<x.y<<")] = "<<F.f(x)<<"\n";
	Vector2F S;
	double a0, a1, a2;
	int i = 0;
	while(i < 2) {

		if(i == 0) {
			S = S0;
			a0 = minimization(F, x, S);
			x = x + S * a0;
			
			S.verbose("S0");
			std::cout<<" a0 "<<a0<<"\n";
			std::cout<<"f[("<<x.x<<" , "<<x.y<<")] = "<<F.f(x)<<"\n";
			
		}
		else if(i==1) {
			S = S1;
			a1 = minimization(F, x, S);
			x = x + S * a1;
			
			S.verbose("S1");
			std::cout<<" a1 "<<a1<<"\n";
			std::cout<<"f[("<<x.x<<" , "<<x.y<<")] = "<<F.f(x)<<"\n";
		}
		i++;
	}

	S = x - x0;
	a2 = minimization(F, x0, S);
	x = x0 + S * a2;

	S.verbose("S");
	std::cout<<" a "<<a2<<"\n";
	std::cout<<"cycle end\n f[("<<x.x<<" , "<<x.y<<")] = "<<F.f(x)<<"\n";;
	
	S0 = S1;
	S1 = S;
}

double PowellMethod::minimization(BaseFunction & F, const Vector2F & at, const Vector2F & along)
{
	m_f = &F;
	m_at = at;
	m_along = along;
	return goldenSectionSearch(-1000, 0, 1000, 10e-7);
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
	if(tx == tb) {
		//std::cout<<"taxiCab(x) == taxiCab(b)\n";
		return x;
	}
    if (tx < tb) {
      if (c - b > b - a) return goldenSectionSearch(b, x, c, tau);
      else return goldenSectionSearch(a, x, b, tau);
    }
    else {
      if (c - b > b - a) return goldenSectionSearch(a, b, x, tau);
      else return goldenSectionSearch(x, b, c, tau);
    }
}