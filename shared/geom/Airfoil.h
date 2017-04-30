/*
 *  Airfoil.h
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
/// c maximum chord (airfoil length) in meters
/// m maximum camber in meters
/// p position of maximum camber in percentages
/// t maximum thickness in meters

	float m_c, m_m, m_p, m_t;
 
public:
	Airfoil();
/// m in percentages
/// p in tenths [0,9]
/// t1 in tenths [0,9]
/// t2 in percentages [0,9]
	Airfoil(const float & c,
			const int & m,
			const int & p,
			const int & t1,
			const int & t2);
/// m relative to c [0,1]
/// p [0,1]
/// t [0,1]
	Airfoil(const float & c,
			const float & m,
			const float & p,
			const float & t);
	virtual ~Airfoil();
	
	void setChord(const float & c);
/// m maximum camber in percentage of the chord
/// p position of maximum camber in tenths of the chord
/// t1t2 maximum thickness of the airfoil in percentage of chord
	void set4Digit(const int & m,
					const int & p,
					const int &t1,
					const int & t2);
/// m relative to c [0,1]
/// p [0,1]
/// t [0,1]					
	void setCMPT(const float & c,
			const float & m,
			const float & p,
			const float & t);
		
	const float & chord() const;
/// maximum camber in percentage of the chord
	const float camberRatio() const;
	const float & position() const;
	
/// mean camber line coordinate
/// x [0, 1]
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