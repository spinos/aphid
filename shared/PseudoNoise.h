/*
 *  PseudoNoise.h
 *  hair
 *
 *  Created by jian zhang on 5/13/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef _Pseudo_NOISE_H
#define _Pseudo_NOISE_H

class PseudoNoise
{
public:
	PseudoNoise();
	~PseudoNoise() {}
	
	float rfloat(int i);
	int rint1(int i, int mmn = 1000);
	int rint2(int i, int j, int mmn = 1000);
	int rint3(int i, int j, int k, int mmn = 1000);
	void sphereRand(float& x, float& y, float& z, float r, unsigned int &i);
private:
	static unsigned xRand, yRand, zRand;     /* seed */
	void seedd(unsigned int nx, unsigned int ny, unsigned int nz);
	unsigned randomize();
};

#endif
