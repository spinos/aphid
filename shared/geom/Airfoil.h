/*
 *  Airfoil.h
 *  proxyPaint
 *
 *  Created by jian zhang on 12/31/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_GEOM_AIRFOIL_H
#define APH_GEOM_AIRFOIL_H

namespace aphid {

class Airfoil {

/// NACA Four-Digit Series
/// c maximum chord (airfoil length)
/// m maximum camber
/// p position of maximum camber
/// t maximum thickness

	float m_c, m_m, m_p, m_t;
 
public:
	Airfoil();
	Airfoil(const float & c,
			const int & m,
			const int & p,
			const int & t1,
			const int & t2);
	virtual ~Airfoil();
	
	void setChord(const float & c);
/// m maximum camber in percentage of the chord
/// p position of maximum camber in tenths of the chord
/// t1t2 maximum thickness of the airfoil in percentage of chord
	void set4Digit(const int & m,
					const int & p,
					const int &t1,
					const int & t2);
		
	const float & chord() const;
	
/// mean camber line coordinate
/// x [0, c]
	float calcYc(const float & x) const;
/// thickness distribution above and below the camber line	
/// x [0, 1]
	float calcYt(const float & x) const;
/// angle of tilting
/// x [0, c]
	float calcTheta(const float & x) const;
	
protected:

};

}

#endif