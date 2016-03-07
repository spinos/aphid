/*
 *  TreeProperty.h
 *  testntree
 *
 *  Created by jian zhang on 3/8/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <string>

namespace aphid {

class TreeProperty {

	int m_maxLevel;
	int m_numEmptyNodes;
	int m_numInternalNodes;
	int m_numLeafNodes;
	int m_minPrims, m_maxPrims, m_totalNPrim;
	float m_emptyVolume;
	float m_totalVolume;
	
public:
	TreeProperty();
	virtual ~TreeProperty();
	void addMaxLevel(int x);
	void addEmptyVolume(float x);
	void setTotalVolume(float x);
	void addNInternal();
	void addNLeaf();
	void updateNPrim(int x);
	
protected:
	void resetPropery();
	std::string logProperty() const;
	
};

}