#ifndef LFMACHINE_H
#define LFMACHINE_H

#include "LfParameter.h"

namespace lfr {
    
class LfMachine  {

	LfParameter * m_param;
    
public:

	LfMachine(LfParameter * param);
	virtual ~LfMachine();
	
	const LfParameter * param() const;
	
	virtual void initDictionary();
	virtual void dictionaryAsImage(unsigned * imageBits, int imageW, int imageH);
	virtual void fillSparsityGraph(unsigned * imageBits, int iLine, int imageW, unsigned fillColor);
	virtual void preLearn();
	virtual void learn(const ExrImage * image, int iPatch0, int iPatch1);
	virtual void updateDictionary(const ExrImage * image, int t);
	virtual void cleanDictionary();
	virtual void recycleData();
	virtual float computePSNR(const ExrImage * image, int iImage);
	
protected:
	void fillPatch(unsigned * dst, float * color, int s, int imageW, int rank = 3);
	LfParameter * param1();
	
private:
	
};

}
#endif        //  #ifndef LFMACHINE_H

