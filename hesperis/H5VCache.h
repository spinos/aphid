/*
 *  H5VCache.h
 *  opium
 *
 *  vertex data reader and blender
 *
 *  Created by jian zhang on 3/2/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_H5_VCACHE_H
#define APH_H5_VCACHE_H

#include <H5Holder.h>
#include <ASampleTime.h>

namespace aphid {

class BaseBuffer;

class H5VCache : public H5Holder
{
/// linear interpolation
	aphid::BaseBuffer * m_data[2];
/// number of points per piece
    aphid::BaseBuffer * m_counts;
/// piece begins
    aphid::BaseBuffer * m_drifts;
/// trigger of read data
    unsigned m_minInd;
    int m_numPieces;
    int m_numPoints;
	bool m_hasData;
    bool m_hasPiecesChecked;
    bool m_hasArbitrarySampleChecked;
	float m_blender;
	    bool m_hasArbitrarySamples;
/// frame.subframe, subframe > 99
    ASampleTimeI m_arbitrarySamples;

public:
	H5VCache();
	virtual ~H5VCache();
	
protected:
	void resetMinInd();
	void setMinInd(int x);
	bool isFirstPiece(int x) const;
	void checkPieces(const std::string & pathName, unsigned numIters);
	int driftIndex(int idx) const;
	char readFrame(float *data, int count, 
				const char *h5name, int frame, int sample,
				bool isLegacy);
	char readFrameLegacy(float *data, int count, 
				const char *h5name, int frame, int sample);
	void mixFrames(float weight0, float weights1);
    
	const bool & hasData() const;
	const bool & hasPiecesChecked() const;
	const bool& hasArbitrarySampleChecked() const;
	const bool& hasArbitrarySamples() const;
	const int & numPieces() const;
	const int & numPoints() const;

	bool readData(const std::string & fileName, 
                   const std::string & pathName,
                   double dtime,
                   unsigned numIterPoints,
                   int isSubframe,
				   bool toSkipReading,
				   bool isLegacy);
	float * data0() const;
	
	void mixData(const H5VCache * b, float alpha);
	
	bool isBlenderChanged(float x) const;
	void setBlender(float x);
	
	void checkArbitrarySamples(const std::string & pathName);
	
private:
	void scanExclusive(int n, int * dst, int * src);
	bool asSampleTime(int& frame, int& subframe, const std::string& stime) const;
	void addASample(const std::string & stime);
	bool findArbitrarySample(const double& dtime);
	void setSampleWeight1(const int& frame,
                const int& sample0, const int& sample1,
                const int& samplex);
   void setSampleWeight2(const int& frame,
                const int& sample1,
                const int& samplex);
};

}
#endif
