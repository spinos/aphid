/*
 *  Vegetation.h
 *  garden
 *
 *  collection of patches at different angles
 *
 *  Created by jian zhang on 4/26/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_VEGETATION_H
#define GAR_VEGETATION_H

class VegetationPatch;

class Vegetation {

#define NUM_ANGLE 11
#define NUM_VARIA 11
#define TOTAL_NUM_P 121
	VegetationPatch * m_patches[TOTAL_NUM_P];
	int m_numPatches;
	
public:
	Vegetation();
	virtual ~Vegetation();
	
	VegetationPatch * patch(const int & i);
	
	void setNumPatches(int x);
	const int & numPatches() const;
	
	int getMaxNumPatches() const;
	
	void rearrange();
	
protected:

private:
};

#endif