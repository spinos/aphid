#include "HField.h"
#include <AllHdf.h>
#include <SHelper.h>
#include <AField.h>
#include <BaseBuffer.h>

HField::HField(const std::string & path) : HBase(path) {}
HField::~HField() {}
	
char HField::verifyType() 
{
    if(!hasNamedAttr(".fieldNumChannels"))
        return 0;
    
    if(!hasNamedAttr(".fieldChannelNames"))
        return 0;
    
    return 1;
}

char HField::save(AField * fld) 
{
	if(fld->numChannels() < 1) {
		std::cout<<"\n field has no channel";
		return 0;
	}
	
    std::vector<std::string > names;
    fld->getChannelNames(names);
    
	int nc = names.size();
    if(!hasNamedAttr(".fieldNumChannels"))
		addIntAttr(".fieldNumChannels");
	
	writeIntAttr(".fieldNumChannels", &nc);
	
	std::string combined = SHelper::Combine(names);
    
    if(!hasNamedAttr(".fieldChannelNames"))
		addStringAttr(".fieldChannelNames", combined.size());
		
	writeStringAttr(".fieldChannelNames", combined);
	
	std::vector<std::string >::const_iterator it = names.begin();
	for(; it!= names.end();++it) saveAChannel(*it, fld->namedChannel(*it));
	
    return 1;
}

char HField::load(AField * fld) 
{
    if(!verifyType()) return 0;
    
	int nc = 1;
    readIntAttr(".fieldNumChannels", &nc);
	
	std::string combined;
	readStringAttr(".fieldChannelNames", combined);
	std::vector<std::string > channelNames;
	SHelper::Split(combined, channelNames);
	
	if(channelNames.size() != nc) {
		std::cout<<"/n n channel names not match";
		return 0;
	}
	
	std::vector<std::string >::const_iterator it = channelNames.begin();
	for(; it!= channelNames.end();++it) loadAChannel(*it, fld);
	
    return 1;
}

void HField::saveAChannel(const std::string& name, TypedBuffer * chan)
{
	HBase grp(childPath(name));
	int ne = chan->numElements();
	
    if(!grp.hasNamedAttr(".nelm"))
		grp.addIntAttr(".nelm");
	
	grp.writeIntAttr(".nelm", &ne);
	
	if(!grp.hasNamedAttr(".typ"))
		grp.addIntAttr(".typ");
		
	int t = chan->valueType();
	grp.writeIntAttr(".typ", &t);
	
	if(chan->valueType() == TypedBuffer::TFlt) {
		if(!grp.hasNamedData(".def"))
			grp.addFloatData(".def", ne);
			
		grp.writeFloatData(".def", ne, chan->typedData<float>());
	}
	else if(chan->valueType() == TypedBuffer::TVec3) {
		if(!grp.hasNamedData(".def"))
			grp.addVector3Data(".def", ne);
			
		grp.writeVector3Data(".def", ne, chan->typedData<Vector3F>());
	}
	
	grp.close();
    
    HBase gbake(grp.childPath(".bake"));
    gbake.close();
}

void HField::loadAChannel(const std::string& name, AField * fld)
{
	if(!hasNamedChild(name.c_str())) {
		std::cout<<"\n has no child "<<name;
		return;
	}
	
	HBase grp(childPath(name));
	int ne = 1;
	grp.readIntAttr(".nelm", &ne);
	
	int t = 1;
	grp.readIntAttr(".typ", &t);
	
	if(t==TypedBuffer::TFlt) {
		fld->addFloatChannel(name, ne);
	}
	else if(t==TypedBuffer::TVec3) {
		fld->addVec3Channel(name, ne);
	}
	
	TypedBuffer * chan = fld->namedChannel(name);
	
	if(t==TypedBuffer::TFlt) {
		grp.readFloatData(".def", ne, chan->typedData<float>());
	}
	else if(t==TypedBuffer::TVec3) {
		grp.readVector3Data(".def", ne, chan->typedData<Vector3F>());
	}
		
	grp.close();
}

void HField::saveFrame(const std::string & frame, AField * fld)
{
    std::vector<std::string > names;
    fld->getChannelNames(names);
    
    std::vector<std::string >::const_iterator it = names.begin();
	for(; it!= names.end();++it) 
        saveAChannelFrame(frame, *it, fld->namedChannel(*it));
}

void HField::saveAChannelFrame(const std::string & frame,
                           const std::string& channelName, TypedBuffer * chan)
{
    HBase grp(childPath(channelName));
	int ne = chan->numElements();
    HBase gbake(grp.childPath(".bake"));
    
    if(chan->valueType() == TypedBuffer::TFlt) {
		if(!gbake.hasNamedData(frame.c_str()))
			gbake.addFloatData(frame.c_str(), ne);
			
		gbake.writeFloatData(frame.c_str(), ne, chan->typedData<float>());
	}
	else if(chan->valueType() == TypedBuffer::TVec3) {
		if(!gbake.hasNamedData(frame.c_str()))
			gbake.addVector3Data(frame.c_str(), ne);
			
		gbake.writeVector3Data(frame.c_str(), ne, chan->typedData<Vector3F>());
	}
    
    gbake.close();
    grp.close();
}
//:~