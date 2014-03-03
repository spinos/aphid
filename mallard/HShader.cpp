/*
 *  HShader.cpp
 *  mallard
 *
 *  Created by jian zhang on 1/11/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "HShader.h"
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include "FeatherShader.h"

HShader::HShader(const std::string & path) : HBase(path) {}
	
char HShader::save(ShaderGroup * g)
{
	int ns = g->numShaders();
	
	if(!hasNamedAttr(".ns"))
		addIntAttr(".ns");
		
	writeIntAttr(".ns", &ns);
	std::cout<<" num shaders "<<ns<<"\n";
	
	if(ns < 1) return 0;

	for(int i = 0; i < ns; i++)
		writeShader(g->getShader(i));
	
	return 1;
}

char HShader::load(ShaderGroup * g)
{
	if(!hasNamedAttr(".ns")) {
		std::cout<<"no ns";
		return 0;
	}
	
	int ns = numChildren();
	readIntAttr(".ns", &ns);
	std::cout<<" num shaders "<<ns<<"\n";
	
	for(int i = 0; i < ns; i++) {
		HBase c((boost::format("%1%/%2%") % pathToObject() % getChildName(i)).str());
		readShader(&c, g);
		c.close();
	}
	
	return 1;
}

void HShader::writeShader(BaseShader * s) 
{
    HBase g((boost::format("%1%/%2%") % pathToObject() % s->name()).str());
	
    switch(s->shaderType()) {
        case BaseShader::TFeather:
            writeFeatherShader(s, &g);
            break;
        default:
            break;
    }
}

void HShader::writeFeatherShader(BaseShader * s, HBase * g)
{
    if(!g->hasNamedAttr(".typ")) g->addIntAttr(".typ");
    int typ = 0;
	g->writeIntAttr(".typ", (int *)(&typ));
	
    FeatherShader * f = static_cast<FeatherShader *>(s);
    
    if(!g->hasNamedAttr(".gloss")) g->addFloatAttr(".gloss");
	float gloss = f->gloss();
	g->writeFloatAttr(".gloss", &gloss);
	
	if(!g->hasNamedAttr(".gloss2")) g->addFloatAttr(".gloss2");
	float gloss2 = f->gloss2();
	g->writeFloatAttr(".gloss2", &gloss2);
}

void HShader::readShader(HBase * c, ShaderGroup * g) 
{
    std::cout<<boost::format("read %1%\n") % c->pathToObject();
	if(!c->hasNamedAttr(".typ")) return;
	int typ = 0;
	c->readIntAttr(".typ", &typ);
	BaseShader *s = 0;
	switch (typ) {
		case 0:
			s = readFeatherShader(c);
			break;
		default:
			break;
	}
	
	if(!s) return;
	
	std::string name = c->pathToObject();
	boost::erase_first(name, pathToObject());
	boost::erase_first(name, "/");
	s->setName(name);
	
	g->addShader(s);
}

FeatherShader * HShader::readFeatherShader(HBase * b)
{
    FeatherShader * f = new FeatherShader;
    float gloss = 1.f;
	if(b->hasNamedAttr(".gloss")) 
		b->readFloatAttr(".gloss", &gloss);
	f->setGloss(gloss);
	
	float gloss2 = 1.f;
	if(b->hasNamedAttr(".gloss2")) 
		b->readFloatAttr(".gloss2", &gloss2);
	f->setGloss2(gloss2);
    return f;
}

