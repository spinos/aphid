#include "HFrameRange.h"
#include "AFrameRange.h"

namespace aphid {

HFrameRange::HFrameRange(const std::string & path) : HBase(path) 
{}

HFrameRange::~HFrameRange()
{}
	
char HFrameRange::verifyType()
{
    if(!hasNamedAttr(".sf"))
		return 0;
	if(!hasNamedAttr(".ef"))
		return 0;
	if(!hasNamedAttr(".spf"))
		return 0;
    return 1;
}

char HFrameRange::save(AFrameRange * tm)
{
    if(!hasNamedAttr(".sf"))
        addIntAttr(".sf");
    
    writeIntAttr(".sf", &tm->FirstFrame);
	
	if(!hasNamedAttr(".ef"))
        addIntAttr(".ef");
    
    writeIntAttr(".ef", &tm->LastFrame);
	
	if(!hasNamedAttr(".spf"))
        addIntAttr(".spf");
    
    writeIntAttr(".spf", &tm->SamplesPerFrame);
    
    if(!hasNamedAttr(".fps"))
        addFloatAttr(".fps");
    
    writeFloatAttr(".fps", &tm->FramesPerSecond);
    
    if(tm->segmentExpr().size() > 3) {
        if(!hasNamedAttr(".sgxr"))
            addVLStringAttr(".sgxr");
        
        writeVLStringAttr(".sgxr", tm->segmentExpr() );
    }
    return 1;
}

char HFrameRange::load(AFrameRange * tm)
{
	readIntAttr(".sf", &tm->FirstFrame);
	readIntAttr(".ef", &tm->LastFrame);
	readIntAttr(".spf", &tm->SamplesPerFrame);
    readFloatAttr(".fps", &tm->FramesPerSecond);
    if(hasNamedAttr(".sgxr"))
        readVLStringAttr(".sgxr", tm->segmentExprRef() );
    return 1;
}

}