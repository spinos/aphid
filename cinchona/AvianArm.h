/*
 *  AvianArm.h
 *  cinchona
 *
 *  Created by jian zhang on 1/1/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef AVIAN_ARM_H
#define AVIAN_ARM_H

#include <vector>

namespace aphid {

class Vector3F;
class Matrix44F;

template<typename T1, typename T2>
class HermiteInterpolatePiecewise;

}

class Ligament;
class FeatherObject;
class FeatherOrientationParam;
class FeatherGeomParam;
class FeatherDeformParam;
class Geom1LineParam;
class WingRib;
class WingSpar;

class AvianArm {

	FeatherOrientationParam * m_orientationParam;
	FeatherGeomParam * m_featherGeomParam;
	FeatherDeformParam * m_featherDeformParam;
	WingRib * m_ribs[5];
/// 2 upper 2 lower
	WingSpar * m_spars[4];
	bool m_starboard;
	
#define NUM_MAT 14
/// in order
/// humerus
/// ulna
/// radius
/// carpus
/// second_digit
/// principle
/// inverse_principle
/// hand
/// inverse_hand
/// finger
/// inverse_finger
/// inboard 
/// mid_section_0
/// mid_section_1
	aphid::Matrix44F * m_skeletonMatrices;
/// shoulder-wrist-2nd_digit-2nd_digit_end
	Ligament * m_leadingLigament;
/// shoulder-elbow-wrist-2nd_digit_end
	Ligament * m_trailingLigament;
	float m_secondDigitLength;
	std::vector<FeatherObject *> m_feathers;
	
public:
	AvianArm();
	virtual ~AvianArm();
	
/// idx-th skeleton matrix
	const aphid::Matrix44F & skeletonMatrix(const int & idx) const;
/// shoulder to wrist
	bool updatePrincipleMatrix();
	bool updateHandMatrix();
	bool updateFingerMatrix();
	void updateLigaments();
	
protected:
	aphid::Matrix44F * skeletonMatricesR();
	aphid::Matrix44F * principleMatrixR();
	aphid::Matrix44F * invPrincipleMatrixR();
	aphid::Matrix44F * handMatrixR();
	aphid::Matrix44F * invHandMatrixR();
	aphid::Matrix44F * fingerMatrixR();
	aphid::Matrix44F * invFingerMatrixR();
	aphid::Matrix44F * secondDigitMatirxR();
	aphid::Matrix44F * inboardMarixR();
	aphid::Matrix44F * midsection0MarixR();
	aphid::Matrix44F * midsection1MarixR();
	aphid::Matrix44F * radiusMatrixR();
	
	aphid::Vector3F shoulderPosition() const;
	aphid::Vector3F elbowPosition() const;
	aphid::Vector3F wristPosition() const;
	aphid::Vector3F secondDigitPosition() const;
	aphid::Vector3F secondDigitEndPosition() const;
	
	void set2ndDigitLength(const float & x);
	void setLeadingLigamentOffset(const int & idx,
							const aphid::Vector3F & v);
	void setTrailingLigamentOffset(const int & idx,
							const aphid::Vector3F & v);
	void setLeadingLigamentTangent(const int & idx,
							const aphid::Vector3F & v);
	void setTrailingLigamentTangent(const int & idx,
							const aphid::Vector3F & v);	
	
	const Ligament & leadingLigament() const;
	const Ligament & trailingLigament() const;
	
	Ligament * leadingLigamentR();
	
	FeatherGeomParam * featherGeomParameter();
	bool isFeatherGeomParameterChanged() const;
	bool isFeatherOrientationChanged() const;
	void updateFeatherGeom();
	void updateFeatherTransform();
	void updateRibs();
	void updateSpars();
	
	int numFeathers() const;
	const FeatherObject * feather(int i) const;
	
	FeatherOrientationParam * orientationParameter();
	FeatherDeformParam * featherDeformParameter();
	void updateFeatherDeformation();
	
/// i-th rib
	const WingRib * rib(int i) const;
/// 0:1 upper 2:3 lower
	const WingSpar * spar(int i) const;
	
	void setStarboard(bool x);
	const bool & isStarboard() const;
	
private:
    void clearFeathers();
	void updateFeatherLineRotationOffset(Geom1LineParam * line, 
							const float * yawWeight,
							int & it);
	void updateFeatherLineTranslation(Geom1LineParam * line, 
							const Ligament * lig,
							int & it);
	void updateFeatherLineTranslation(Geom1LineParam * line, 
							const WingSpar * spr,
							int & it);
/// i-th line
	void updateFeatherLineRotation(const int & iline,
							Geom1LineParam * line, 
							int & it);
	void updateFeatherLineGeom(Geom1LineParam * line,
				const aphid::HermiteInterpolatePiecewise<float, aphid::Vector3F > * curve,
				const float & maxC = -1.f);
	void updateWarp(Geom1LineParam * line, 
							int & it);

};
#endif