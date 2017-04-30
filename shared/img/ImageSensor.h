/*
 *  ImageSensor.h
 *  
 *  fill an image by sampling source T
 *  sample array in 2D
 *
 *  Created by jian zhang on 3/17/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_IMG_IMAGE_SENSOR_H
#define APH_IMG_IMAGE_SENSOR_H

#include <math/ATypes.h>
#include <img/BoxSampleProfile.h>
#include <img/ImageSpace.h>

namespace aphid {

namespace img {

template<typename T>
class ImageSensor {

/// left top corner of sample array
	Vector2F m_pntLT;
/// sample point step in horizontal direction
	Vector2F m_vecDu;
/// sample point step in vertical direction
	Vector2F m_vecDv;
/// 1 / num sample points
	float m_uSpacing;
	float m_vSpacing;
	int m_rezU, m_rezV;
	
public:
/// left-top, right-top, left-bottom corner in world space
/// horizontal, vertical num samples 
	ImageSensor(const Vector2F & cornerLT,
				const Vector2F & cornerRT, const int & reslutionU,
				const Vector2F & cornerLB, const int & reslutionV );
/// channel[k]
	void sense(Array3<float> * dst, int k, const T & src) const;
	
	void verbose() const;
	
};

template<typename T>
ImageSensor<T>::ImageSensor(const Vector2F & cornerLT,
				const Vector2F & cornerRT, const int & reslutionU,
				const Vector2F & cornerLB, const int & reslutionV)
{
	m_vecDu = cornerRT - cornerLT;
	m_vecDu /= (float)reslutionU;
	m_vecDv = cornerLB - cornerLT;
	m_vecDv /= (float)reslutionV;
	m_pntLT = cornerLT + (m_vecDu * 0.5f + m_vecDv * 0.5f);
	m_rezU = reslutionU;
	m_rezV = reslutionV;
	m_uSpacing = 1.f / (float)reslutionU;
	m_vSpacing = 1.f / (float)reslutionV;
	
}

template<typename T>
void ImageSensor<T>::verbose() const
{	
	std::cout<<"\n ImageSensor resolution ("<<(1.f/m_uSpacing)<<","<<(1.f/m_vSpacing)<<")"
		<<"\n LT "<<m_pntLT
		<<"\n du "<<m_vecDu
		<<"\n dv "<<m_vecDv;
	std::cout.flush();
}

template<typename T>
void ImageSensor<T>::sense(Array3<float> * dst, int k, const T & src) const
{
	img::BoxSampleProfile<float> sampler;
	img::ImageSpace mspace;
	
	sampler._channel = k;
	sampler._defaultValue = 0.5f;
	mspace._sampleSpacing = m_uSpacing;
	
	src.getSampleProfileSapce(&sampler, &mspace);
#if 1
	std::cout<<"\n sampler lo"<<sampler._loLevel
			<<" hi"<<sampler._hiLevel
			<<" mx"<<sampler._mixing
			<<" tm"<<mspace._worldToUVMatrix;
	std::cout.flush();
#endif
	Array2<float> * slice = dst->rank(k);
	
	Vector2F sampnt, sampntj;
	
	for(int j=0;j<m_rezU;++j) {
		float * colj = slice->column(j);
		
		sampntj = m_pntLT + m_vecDu * j;
		
		for(int i=0;i<m_rezV;++i) {
		
			sampnt = sampntj + m_vecDv * i;
			
			mspace.toUV(sampler._uCoord, sampler._vCoord, sampnt);
			
			colj[i] = src.sample(&sampler);
			
		}
	}
}

}

}

#endif