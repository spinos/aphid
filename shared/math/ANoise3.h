/*
 *  ANoise3.h
 *  
 *
 *  Created by jian zhang on 8/9/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_A_NOISE_3_H
#define APH_A_NOISE_3_H

#include <math/Vector3F.h>

namespace aphid {

class ANoise3 {

	static float PermuTable512[512];

public:
	static float SampleGrid(const int & x,
							const int & y,
							const int & z);
	static float Trilinear(const float & x,
							const float & y,
							const float & z);
/* v sample3 in real world
 * o offset3 in [0, 1] avoid sample3 close to zero
 * freq sample3 * frequency
 * lacunarity sample3 * lacunarity after each interation
 * octaves num of interation
 * http://groups.csail.mit.edu/mac/users/rauch/lacunarity/lacunarity.html
 */
	static float FractalF(const float * v,
						const float * o,
						const float & freq,
						const float & lacunarity = 2.f,
						const int & octaves = 5,
						const float & gain = .5f);
/// mimic http://taukeke.com/2015/05/noise-in-nuke/
/// scale x/ysize
/// motion z
/// gain added each level 
	static float Fbm(const float * v,
						const float * o,
						const float & freq,
						const int & octaves = 5,
						const float & lacunarity = 2.f,
						const float & gain = .5f);
	
};

class ANoise3Sampler {

public:
	Vector3F m_noiseOrigin;
	float m_noiseFrequency;
	float m_noiseLacunarity;
	float m_noiseLevel;
	float m_noiseGain;
	int m_noiseOctave; 
	
	ANoise3Sampler();
	
	float sampleNoise3(const float * v) const;
	
};

}
#endif