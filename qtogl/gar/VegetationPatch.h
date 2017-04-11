/*
 *  VegetationPatch.h
 *  garden
 *
 *
 *  Created by jian zhang on 4/19/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_VEGETATION_PATCH_H
#define GAR_VEGETATION_PATCH_H

#include <vector>

class PlantPiece;

class VegetationPatch {

	typedef std::vector<PlantPiece *> PlantListTyp;
/// to roots
	PlantListTyp m_plants;
	
public:
	VegetationPatch();
	virtual ~VegetationPatch();
	
	int numPlants() const;
	const PlantPiece * plant(const int & i) const;
	
	void addPlant(PlantPiece * pl);
	void clearPlants();
	
protected:

private:
};

#endif