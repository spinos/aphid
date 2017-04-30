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

/// world to texcoord
	Matrix44F m_sampleSpace;
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
	ImageSensor(const Vector2F & cornerLT,
				const Vector2F & cornerRT, const int & reslutionU,
				const Vector2F & cornerLB, const int & reslutionV,
				const Vector2F & uvLT = Vector2F(0.f, 0.f),
				const Vector2F & uvRB = Vector2F(1.f, 1.f) );
/// channel[k]
	void sense(Array3<float> * dst, int k, const T & src) const;
	
	void verbose() const;
	
};

template<typename T>
ImageSensor<T>::ImageSensor(const Vector2F & cornerLT,
				const Vector2F & cornerRT, const int & reslutionU,
				const Vector2F & cornerLB, const int & reslutionV,
				const Vector2F & uvLT, const Vector2F & uvRB)
{
	m_vecDu = cornerRT - cornerLT;
	m_vecDu /= (float)reslutionU;
	m_vecDv = cornerLB - cornerLT;
	m_vecDv /= (float)reslutionV;
	m_uSpacing = (uvRB.x - uvLT.x) / (float)reslutionU;
	m_vSpacing = (uvRB.y - uvLT.y) / (float)reslutionV;
	m_pntLT = cornerLT + (m_vecDu * 0.5f + m_vecDv * 0.5f);
	m_rezU = reslutionU;
	m_rezV = reslutionV;
	
	m_sampleSpace.setIdentity();
	Vector3F freq(m_uSpacing / m_vecDu.length(), 
				m_vSpacing / m_vecDv.length(), 
				1.f);				
	m_sampleSpace.scaleBy(freq);
	
	Vector2F q = m_sampleSpace.transform(cornerLT);
	m_sampleSpace.setTranslation(Vector3F(uvLT.x - q.x, uvLT.y - q.y, 0.f));
	
}

template<typename T>
void ImageSensor<T>::verbose() const
{	
	std::cout<<"\n ImageSensor resolution ("<<(1.f/m_uSpacing)<<","<<(1.f/m_vSpacing)<<")"
		<<"\n LT "<<m_pntLT
		<<"\n du "<<m_vecDu
		<<"\n dv "<<m_vecDv
		<<"\n space "<<m_sampleSpace;
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
	
	mspace._worldToUVMatrix = m_sampleSpace;
	
	src.getSampleProfile(&sampler, &mspace);
#if 0
	std::cout<<"\n sampler lo"<<sampler._loLevel
			<<" hi"<<sampler._hiLevel
			<<" mx"<<sampler._mixing;
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