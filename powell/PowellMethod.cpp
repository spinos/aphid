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
	std::cout<<"f[("<<x.x<<" , "<<x.y<<")] = "<<F.f(x)<<"\n";
	Vector2F x0 = x;
	Vector2F S0(1, 0), S1(0, 1);
	float y = F.f(x);
	float y0 = y;
	float Dx, Dy;
	for(int i = 0; i < 15; i++) {
		std::cout<<"i="<<i<<"\n";
		cycle(F, x, S0, S1);
		
		Dx = (x.x - x0.x) * (x.x - x0.x) + (x.y - x0.y) * (x.y - x0.y);
		
		if(Dx < 10e-11) break;
		
		y = F.f(x);
		
		Dy = (y - y0) * (y - y0);
		
		if(Dy < 10e-11) break;
		
		std::cout<<"cycle end\n";
		
		y0 = y;
		x0 = x;
	}
	x.verbose("minimization");
}

void PowellMethod::cycle(BaseFunction & F, Vector2F & x, Vector2F & S0, Vector2F & S1)
{
	Vector2F S, S11;
	float a0, a1, a2;
	int i = 0;
	while(i < 3) {

		if(i == 0) {
			S = S0;
			S.verbose("S0");
			a0 = F.minimization(x, S);
			std::cout<<" a0 "<<a0<<"\n";
			x = x + S * a0;
		}
		else if(i==1) {
			S = S11 = S1;
			S.verbose("S1");
			a1 = F.minimization(x, S);
			std::cout<<" a1 "<<a1<<"\n";
			x = x + S * a1;
		}
		else {
			S.x = a0 * S0.x + a1 * S0.x;
			S.y = a0 * S0.y + a1 * S1.x;
			S.verbose("S2");
			a2 = F.minimization(x, S);
			
			std::cout<<" a2 "<<a2<<"\n";
			x = x + S * a2;
		}
		
		
		std::cout<<"f[("<<x.x<<" , "<<x.y<<")] = "<<F.f(x)<<"\n";
		
		i++;
	}
	
	S0 = S11;
	S1 = S;
	return;
}
