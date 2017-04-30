/*
 *  ExampleWorks.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 3/1/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "ExampleWorks.h"
#include <AllMama.h>
#include <boost/regex.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

using namespace aphid;

ProxyViz * ExampleWorks::PtrViz = NULL;
MObject ExampleWorks::ObjViz = MObject::kNullObj;

ExampleWorks::ExampleWorks()
{}

ExampleWorks::~ExampleWorks()
{}

bool ExampleWorks::validateSelection()
{
	MSelectionList sels;
 	MGlobal::getActiveSelectionList( sels );
	if(!validateViz(sels)) {
	    MGlobal::displayWarning("no proxyViz selected");
		return 0;
	}
	return 1;
}

bool ExampleWorks::validateViz(const MSelectionList &sels)
{
	PtrViz = NULL;
	ObjViz = MObject::kNullObj;
	
	if(sels.length() < 1) {
		return 0;
	}
			
    MStatus stat;
    MItSelectionList iter(sels, MFn::kPluginLocatorNode, &stat );
    
	iter.getDependNode(ObjViz);
    if(ObjViz == MObject::kNullObj) {
		return 0;
	}
        
	MFnDependencyNode fviz(ObjViz);
	if(fviz.typeName() != "proxyViz") {
		return 0;
	}
	
	PtrViz = (ProxyViz*)fviz.userNode();
    if(!PtrViz) {
        return 0;
	}
    
    return 1;
}

void ExampleWorks::getConnectExamples(MObjectArray & exmpOs)
{
	MPlug dstPlug;
	AHelper::getNamedPlug(dstPlug, ObjViz, "inExample");
	
	MPlugArray srcPlugs;
	ConnectionHelper::GetArrayPlugInputConnections(srcPlugs, dstPlug);
	
	const int n = srcPlugs.length();
	for(int i=0;i<n;++i) {
		MObject exmpObj = srcPlugs[i].node();
		exmpOs.append(exmpObj);
	}
}

MString ExampleWorks::getExampleStatusStr()
{
	validateSelection();
	if(ObjViz == MObject::kNullObj) {
        return "none"; 
	}
	
	MString res("/.name/default");
	MFnDependencyNode fviz(ObjViz);
	bool isActive = true;
	bool isVisible = true;
	
	AttributeHelper::getBoolAttributeByName(fviz, "exampleActive", isActive);
	addBoolStatusStrSeg(res, isActive, "/.is_active");
	
	AttributeHelper::getBoolAttributeByName(fviz, "exampleVisible", isVisible);
	addBoolStatusStrSeg(res, isVisible, "/.is_visible");
	
	double r, g, b;
	AttributeHelper::getColorAttributeByName(fviz, "dspColor", r, g, b);
	addVec3StatusStrSeg(res, r, g, b, "/.dsp_color");
	
	int priority = 1;
	AttributeHelper::getIntAttributeByName(fviz, "examplePriority", priority);
	addIntStatusStrSeg(res, priority, "/.priority");
	
	MObjectArray exmpOs;
	getConnectExamples(exmpOs);
	
	const int n = exmpOs.length();
	
	for(int i=0;i<n;++i) {
		MFnDependencyNode fexmp(exmpOs[i]);
		res += "/.name/";
		res += fexmp.name();
		
		AttributeHelper::getBoolAttributeByName(fexmp, "exampleActive", isActive);
		addBoolStatusStrSeg(res, isActive, "/.is_active");
		
		AttributeHelper::getBoolAttributeByName(fexmp, "exampleVisible", isVisible);
		addBoolStatusStrSeg(res, isVisible, "/.is_visible");
		
		AttributeHelper::getColorAttributeByName(fexmp, "dspColor", r, g, b);
		addVec3StatusStrSeg(res, r, g, b, "/.dsp_color");
	
		AttributeHelper::getIntAttributeByName(fexmp, "examplePriority", priority);
		addIntStatusStrSeg(res, priority, "/.priority");
	}
	
	return res;
}

void ExampleWorks::addBoolStatusStrSeg(MString & res, bool b, const char * segName)
{
	res += segName;
	if(b) {
		res += "/on";
	} else {
		res += "/off";
	}
}

void ExampleWorks::addVec3StatusStrSeg(MString & res, double r, double g, double b, const char * segName)
{
	res += segName;
	res += str(boost::format("/%1% %2% %3%") % r % g % b).c_str();
}

void ExampleWorks::addIntStatusStrSeg(MString & res, int b, const char * segName)
{
	res += segName;
	res += str(boost::format("/%1%") % b).c_str();
}

void ExampleWorks::processShowVoxelThreshold(float x)
{
	validateSelection();
	if(ObjViz == MObject::kNullObj) {
		return;
	}
	MFnDependencyNode fviz(ObjViz);
	MPlug psho = fviz.findPlug("svt");
	psho.setValue(x);
}

float ExampleWorks::getShowVoxelThreshold()
{
	float r = 1.f;
	validateSelection();
	if(ObjViz != MObject::kNullObj) {
		MFnDependencyNode fviz(ObjViz);
		MPlug psho = fviz.findPlug("svt");
		psho.getValue(r);
	}
	return r;
}

bool ExampleWorks::setExampleStatus(int idx, const std::string & expression)
{
	validateSelection();
	if(ObjViz == MObject::kNullObj) {
		return false;
	}
	
	MObject oexmp;
	
	if(idx < 1) {
		oexmp = ObjViz;
	} else {
		MObjectArray exmpOs;
		getConnectExamples(exmpOs);
		if(idx-1 < exmpOs.length() ) {
			oexmp = exmpOs[idx-1];
		}
	}
	
	if(oexmp == MObject::kNullObj) {
		return false;
	}
	
	const std::string pattern = "([a-z_0-9]+=[a-z0-9 -.]+;)";
	
	std::string::const_iterator start, end;
    start = expression.begin();
    end = expression.end();
	
	const boost::regex re1(pattern);
	boost::match_results<std::string::const_iterator> what;
	
	while(regex_search(start, end, what, re1, boost::match_default) ) {
	
		setExampleStatus(oexmp, what[0]);
		
		start = what[0].second;
	}
	
	return true;
}

void ExampleWorks::setExampleStatus(const MObject & exmpO,
				const std::string & expression)
{	
///	std::cout<<"\n ExampleWorks::setExampleStatus "<<expression;
	
	const std::string pattern = "([a-z_0-9]+)=([a-z0-9 -.]+);";
	
	std::string::const_iterator start, end;
    start = expression.begin();
    end = expression.end();
	
	const boost::regex re1(pattern);
	boost::match_results<std::string::const_iterator> what;
	while(regex_search(start, end, what, re1, boost::match_default) ) {
		
		std::string shead;
		std::string sval;
		
		for(unsigned i = 0; i <what.size(); ++i) {
		    if(i==1) {
				shead = what[1];
			} else if(i==2) {
				sval = what[2];
			}
		}
		
		if(shead == "active") {
			setExampleBoolAttrib(exmpO, "exampleActive", sval);
		}
		if(shead == "visible") {
			setExampleBoolAttrib(exmpO, "exampleVisible", sval);
		}
		if(shead == "dspcolor") {
			setExampleCol3Attrib(exmpO, "dspColor", sval);
		}
		if(shead == "priority") {
			setExampleIntAttrib(exmpO, "examplePriority", sval);
		}
		
		start = what[0].second;
	}
}

void ExampleWorks::setExampleBoolAttrib(const MObject & exmpO,
				const MString & attribName,
				const std::string & expression)
{
	bool val;
	if(!matchedBool(val, expression)) {
		return;
	}
	
	MStatus stat;
	MFnDependencyNode fexmp(exmpO, &stat);
	if(!stat) {
		return;
	}
	
	fexmp.findPlug(attribName).setValue(val);
}

void ExampleWorks::setExampleCol3Attrib(const MObject & exmpO,
				const MString & attribName,
				const std::string & expression)
{
	float v[3];
	if(!matchedVec3(v, expression)) {
		return;
	}
	
	MStatus stat;
	MFnDependencyNode fexmp(exmpO, &stat);
	if(!stat) {
		return;
	}
	
	fexmp.findPlug(attribName+"R").setValue(v[0]);
	fexmp.findPlug(attribName+"G").setValue(v[1]);
	fexmp.findPlug(attribName+"B").setValue(v[2]);
}

bool ExampleWorks::matchedBool(bool & val,
				const std::string & expression)
{
	const std::string pattern1 = "(on|1|true)";
	const boost::regex re1(pattern1);
	boost::match_results<std::string::const_iterator> what;
	if(regex_match(expression, what, re1, boost::match_default) ) {
		val = true;
		return true;
	}
	
	const std::string pattern2 = "(off|0|false)";
	const boost::regex re2(pattern2);
	
	if(regex_match(expression, what, re2, boost::match_default) ) {
		val = false;
		return true;
	}
	return false;
}

bool ExampleWorks::matchedVec3(float * vs,
				const std::string & expression)
{
	const std::string pattern1 = "(^[+-]?[0-9]*\\.?[0-9]+|[0-9]+\\.?[0-9]*)([eE][+-]?[0-9]+)?";
	
	std::string::const_iterator start, end;
    start = expression.begin();
    end = expression.end();
	
	int i = 0;
	const boost::regex re1(pattern1);
	boost::match_results<std::string::const_iterator> what;
	while(regex_search(start, end, what, re1, boost::match_default) ) {
	
		try {
			vs[i++] = boost::lexical_cast<float>(what[0]);
			
		} catch (boost::bad_lexical_cast &) {
            std::cout<<"\n bad cast "<<what[0];
			return false;
        }
		
		if(i==3) {
			return true;
		}
		start = what[0].second;
	}
	return false;
}

void ExampleWorks::setExampleIntAttrib(const MObject & exmpO,
				const MString & attribName,
				const std::string & expression)
{
	int val;
	if(!matchedInt(&val, expression)) {
		return;
	}
	
	MStatus stat;
	MFnDependencyNode fexmp(exmpO, &stat);
	if(!stat) {
		return;
	}
	
	fexmp.findPlug(attribName).setValue(val);
}

bool ExampleWorks::matchedInt(int * vs,
				const std::string & expression)
{
	const std::string pattern1 = "(^[+-]?[0-9]*)";
	
	std::string::const_iterator start, end;
    start = expression.begin();
    end = expression.end();
	
	int i = 0;
	const boost::regex re1(pattern1);
	boost::match_results<std::string::const_iterator> what;
	while(regex_search(start, end, what, re1, boost::match_default) ) {
	
		try {
			vs[i++] = boost::lexical_cast<int>(what[0]);
			
		} catch (boost::bad_lexical_cast &) {
            std::cout<<"\n bad cast "<<what[0];
			return false;
        }
		
		if(i==1) {
			return true;
		}
		start = what[0].second;
	}
	return false;
}

void ExampleWorks::getVizStatistics(std::map<std::string, std::string > & stats)
{
	validateSelection();
	if(!PtrViz) {
		return;
	}
	PtrViz->getStatistics(stats);
}

bool ExampleWorks::getActiveExamplePriority(std::map<int, int > & stats)
{
	validateSelection();
	if(ObjViz == MObject::kNullObj) {
		return false;
	}
	
	bool isActive = false;
	int prior = 1;
	
	MFnDependencyNode fviz(ObjViz);
	AttributeHelper::getBoolAttributeByName(fviz, "exampleActive", isActive);
	if(isActive) {
		AttributeHelper::getIntAttributeByName(fviz, "examplePriority", prior);
		stats[0] = prior;
	}
	
	MObjectArray exmpOs;
	getConnectExamples(exmpOs);
	
	const int n = exmpOs.length();
	
	for(int i=0;i<n;++i) {
		MFnDependencyNode fexmp(exmpOs[i]);
		AttributeHelper::getBoolAttributeByName(fexmp, "exampleActive", isActive);
		if(isActive) {
			AttributeHelper::getIntAttributeByName(fexmp, "examplePriority", prior);
			stats[i+1] = prior;
		}
	
	}
	return true;
}

void ExampleWorks::getExampleColors(std::vector<Vector3F> & colors)
{
	if(ObjViz == MObject::kNullObj) {
        return; 
	}
	
	MFnDependencyNode fviz(ObjViz);
	double r, g, b;
	AttributeHelper::getColorAttributeByName(fviz, "dspColor", r, g, b);
	
	colors.push_back(Vector3F(r, g, b) );
	
	MObjectArray exmpOs;
	getConnectExamples(exmpOs);
	
	const int n = exmpOs.length();
	
	for(int i=0;i<n;++i) {
		MFnDependencyNode fexmp(exmpOs[i]);
		AttributeHelper::getColorAttributeByName(fexmp, "dspColor", r, g, b);
		colors.push_back(Vector3F(r, g, b) );
	}
}

void ExampleWorks::activateExamples()
{
	std::map<int, int > priotityMap;
	if(!getActiveExamplePriority(priotityMap) ) {
		return;
	}
	
	std::vector<int> inds;
	std::map<int, int >::const_iterator it = priotityMap.begin();
	for(;it!=priotityMap.end();++it) {
		int n = it->second;
		for(int i=0;i<n;++i) {
			inds.push_back(it->first);
		}
	}
	
	if(inds.size() < 1) {
		std::cout<<"\n WARNING no active examples, use default 0";
		inds.push_back(0);
	}
	
	std::vector<Vector3F> colorVec;
	getExampleColors(colorVec);
	
	PtrViz->processFilterPlantTypeMap(inds, colorVec);
}

void ExampleWorks::updateSampleColor()
{
	validateSelection();
	if(!PtrViz) {
		return;
	}
	
	std::vector<Vector3F> colorVec;
	getExampleColors(colorVec);
	PtrViz->processSampleColorChanges(colorVec);
}
//:~