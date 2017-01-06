/*
 *  Ligament.h
 *  cinchona
 *
 *  Created by jian zhang on 1/2/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef LIGAMENT_H
#define LIGAMENT_H

namespace aphid {

class Vector3F;

template<typename T1, typename T2>
class HermiteInterpolatePiecewise;

}

class Ligament {
	
	aphid::HermiteInterpolatePiecewise<float, aphid::Vector3F > * m_interp;
	aphid::Vector3F * m_knotPoint;
	aphid::Vector3F * m_knotOffset;
	aphid::Vector3F * m_knotTangent;
	
public:
	Ligament(const int & np = 3);
	virtual ~Ligament();
	
	void setKnotOffset(const int & idx,
				const aphid::Vector3F & v);
				
	void setKnotPoint(const int & idx,
				const aphid::Vector3F & v);
				
	void setKnotTangent(const int & idx,
				const aphid::Vector3F & v,
				int side=2);
							
	void update();
				
	aphid::Vector3F getPoint(const int & idx,
				const float & param) const;
	
	aphid::Vector3F getDerivative(const int & idx,
				const float & param) const;
	
	const int & numPieces() const;
	
protected:
/// idx-th knot
/// idx = 0: 0-th piece begin
/// idx = n piece: (n - 1)-piece end
/// else (idx-1)-th piece end and idx-th piece begin
	void setKnot(const int & idx,
				const aphid::Vector3F & pt,
				const aphid::Vector3F & tg0,
				const aphid::Vector3F & tg1);
				
private:
};

#endif