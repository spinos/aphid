/*
 *  H5VCache.h
 *  opium
 *
 *  Created by jian zhang on 3/2/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include <H5Holder.h>

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
	const int & numPieces() const;

	bool readData(const std::string & fileName, 
                   const std::string & pathName,
                   double dtime,
                   unsigned numIterPoints,
                   int isSubframe,
				   bool isLegacy = false);
	float * data0();
	
private:
	void scanExclusive(int n, int * dst, int * src);
	
};

}