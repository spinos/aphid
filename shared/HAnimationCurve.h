#ifndef HANIMATIONCURVE_H
#define HANIMATIONCURVE_H

/*
 *  HAnimationCurve.h
 *  aphid
 *
 *  Created by jian zhang on 8/27/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <HBase.h>
class AAnimationKey;
class AAnimationCurve;
class HAnimationCurve : public HBase {
public:
	HAnimationCurve(const std::string & path);
	virtual ~HAnimationCurve();
	
	virtual char verifyType();
    virtual char save(AAnimationCurve * curve);
	virtual char load(AAnimationCurve * curve);
private:
	void saveKeys(AAnimationCurve * curve, int n);
	void loadKeys(AAnimationCurve * curve, int n);
	void saveKey(unsigned i, const AAnimationKey & key);
	AAnimationKey loadKey(int i);
};
#endif        //  #ifndef HANIMATIONCURVE_H
