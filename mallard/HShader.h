/*
 *  HShader.h
 *  mallard
 *
 *  Created by jian zhang on 1/11/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
 
#pragma once

#include <HBase.h>
#include <ShaderGroup.h>
class FeatherShader;
class HShader : public HBase {
public:
	HShader(const std::string & path);
	
	char save(ShaderGroup * g);
	char load(ShaderGroup * g);

protected:
    virtual void writeShader(BaseShader * s);
    virtual void readShader(HBase * b, ShaderGroup * g);
    
private:
    void writeFeatherShader(BaseShader * c, HBase * g);
    FeatherShader * readFeatherShader(HBase * b);
};
