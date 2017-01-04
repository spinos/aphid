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

}

class Ligament;
class FeatherObject;
class FeatherGeomParam;

class AvianArm {

	FeatherGeomParam * m_featherGeomParam;
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
	float * m_featherX;
	
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
	void updateFeatherGeom();
	void updateFeatherTransform();
	
	int numFeathers() const;
	const FeatherObject * feather(int i) const;
	
private:
    void clearFeathers();
	
	
};
#endif