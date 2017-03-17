/*
 *  HeightField.h
 *  
 *
 *  Created by jian zhang on 3/15/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_IMG_HEIGHT_FIELD_H
#define APH_IMG_HEIGHT_FIELD_H

#include <img/GaussianPyramid.h>

namespace aphid {

namespace img {

class HeightField : public GaussianPyramid<float> {

	SignalTyp m_derivative[7];
/// area covered
	Float2 m_range;
/// range_x / num_samples_x
	float m_sampleSize;
	
public:

	struct Profile {
/// height > 0.0 if value > _zeroHeightValue
		float _zeroHeightValue;
/// height = min height if value == 0.0
		float _minHeight2;
/// height = max height if value == 1.0
		float _maxHeight2;
		float _heightRange;
		void valueToHeight(float & inValue) {
			if(inValue >= _zeroHeightValue) {
				inValue = _maxHeight2 * (inValue - _zeroHeightValue);
			} else {
				inValue = _minHeight2 * (_zeroHeightValue - inValue);
			}
		}
		
	};
	
	HeightField();
	virtual ~HeightField();
	
	virtual void create(const SignalTyp & inputSignal);
	
	void setRange(const float & w);

	const SignalTyp & levelDerivative(int level) const;
	
	float sampleHeight(BoxSampleProfile<float> * prof) const;
/// range of change
	Float2 sampleHeightDerivative(BoxSampleProfile<float> * prof) const;
	const float & sampleSize() const;
	void verbose() const;
	
	static Profile GlobalHeightFieldProfile;
/// fields default value at 2^5 - 2^10
	static Array3<float> InitialValues[6];
	
	static void SetGlobalProfile(float zeroValue, float minHeight, float maxHeight);
	
	static const Array3<float> & InitialValueAtLevel(int level);
				
protected:

private:

};

}

}
#endif