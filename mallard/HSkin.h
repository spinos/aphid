/*
 *  HSkin.h
 *  mallard
 *
 *  Created by jian zhang on 10/1/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <HBase.h>
class MlSkin;
class MlCalamusArray;
class HSkin : public HBase {
public:
	HSkin(const std::string & path);
	
	virtual char save(MlSkin * s);
	virtual char load(MlSkin * s);
private:
	struct CalaVer0 {
		unsigned m_faceIdx, m_featherId, m_bufStart;
		float m_patchU, m_patchV, m_rotX, m_rotY, m_scale;
	};
	void readCurData(MlCalamusArray * arr, const unsigned & num);
	void readV0Data(MlCalamusArray * arr, const unsigned & num);
};