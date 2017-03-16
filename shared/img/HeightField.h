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
#include <math/BoundingBox.h>

namespace aphid {

namespace img {

class HeightField : public GaussianPyramid<float> {

	SignalTyp m_derivative[5];
	BoundingBox m_bbox;
	
public:

	struct Profile {
/// height > 0.0 if value > _zeroHeightValue
		float _zeroHeightValue;
/// height = min height if value == 0.0
		float _minHeight2;
/// height = max height if value == 1.0
		float _maxHeight2;
		
		void set(float zeroValue, float minHeight, float maxHeight)
		{
			_zeroHeightValue = zeroValue;
			_minHeight2 = minHeight * 2.f;
			_maxHeight2 = maxHeight * 2.f;
		}
		
		void valueToHeight(float & inValue) {
			if(inValue >= _zeroHeightValue) {
				inValue = _maxHeight2 * (inValue - _zeroHeightValue);
			} else {
				inValue = _minHeight2 * (_zeroHeightValue - inValue);
			}
		}
		
	};
	
	HeightField();
	HeightField(const SignalTyp & inputSignal);
	virtual ~HeightField();
	
	const BoundingBox & calculateBBox();
	const BoundingBox & getBBox() const;
	
	float sampleHeight(BoxSampleProfile<float> * prof) const;
	
	const SignalTyp & levelDerivative(int level) const;
	
	static Profile GlobalHeightFieldProfile;
	
protected:

private:

};

}

}
#endif