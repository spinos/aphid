/*
 *  SampleFrame.h
 *  opium
 *
 *  Created by jian zhang on 12/23/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <string>
class SampleFrame {
public:
	SampleFrame();
	
	void setFirst(const int& iframe, const int& isample, const float& weight);
	void calculateWeights(double frameTime, const int & spf);
	int sampleOfset0() const;
	int sampleOfset1() const;
    std::string str() const;
	int m_minFrame;
	int m_spf;
	int m_frames[2];
	int m_samples[2];
	float m_weights[2];
};
