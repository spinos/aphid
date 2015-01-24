/*
 *  SimpleSystem.h
 *  proof
 *
 *  Created by jian zhang on 1/25/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <AllMath.h>
class SimpleSystem {
public:
	SimpleSystem();
	
	Vector3F * X() const;
	const unsigned numFaceVertices() const;
	unsigned * indices() const;
	
	Vector3F * groundX() const;
	const unsigned numGroundFaceVertices() const;
	unsigned * groundIndices() const;
	
	Vector3F * Vline() const;
	const unsigned numVlineVertices() const;
	unsigned * vlineIndices() const;
	
	void progress();
	
private:
	Vector3F * m_X;
	unsigned * m_indices;
	
	Vector3F * m_V;
	Vector3F * m_Vline;
	unsigned * m_vIndices;
	
	Vector3F * m_groundX;
	unsigned * m_groundIndices;
};