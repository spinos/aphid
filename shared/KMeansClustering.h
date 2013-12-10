/*
 *  KMeansClustering.h
 *  aphid
 *
 *  Created by jian zhang on 12/7/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>

class KMeansClustering {
public:
	KMeansClustering();
	virtual ~KMeansClustering();
	
	void setInitialGuess(unsigned idx, const Vector3F & pos);
	void preAssign();
	void assignToGroup(unsigned idx, const Vector3F & pos);
	float moveCentroids();
	void resetGroup();
	Vector3F groupCenter(unsigned idx) const;
	unsigned K() const;
	char isValid() const;
	unsigned group(unsigned idx) const;
protected:
    virtual void setK(unsigned k);
	virtual void setN(unsigned n);
	void setValid(char val);
	
private:
	Vector3F * m_centroid;
	Vector3F * m_sum;
	unsigned * m_group;
	unsigned m_k, m_n;
	char m_valid;
};