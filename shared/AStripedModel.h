#ifndef ASTRIPEDMODEL_H
#define ASTRIPEDMODEL_H

/*
 *  AStripedModel.h
 *  aphid
 *
 *  Created by jian zhang on 7/6/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
class BaseBuffer;

class AStripedModel {
public:
	AStripedModel();
	virtual ~AStripedModel();
	
	void create(unsigned n);
	
	unsigned * pointDrifts();
	unsigned * indexDrifts();
	const unsigned numStripes() const;
	
private:
	unsigned m_numStripes;
	BaseBuffer * m_pDrift;
	BaseBuffer * m_iDrift;
};

#endif // ASTRIPEDMODEL_H