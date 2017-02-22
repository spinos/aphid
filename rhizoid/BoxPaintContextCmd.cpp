/*
 *  BoxPaintContextCmd.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 2/1/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "BoxPaintContextCmd.h"
#include "boxPaintTool.h"
#define kMinMarginFlag "-mng"
#define kMinMarginFlagLong "-minMargin"
#define kMaxMarginFlag "-mxg"
#define kMaxMarginFlagLong "-maxMargin"
#define kOptFlag "-opt" 
#define kOptFlagLong "-option"
#define kLsegFlag "-brd" 
#define kLsegFlagLong "-brushRadius"
#define kMinFlag "-smn" 
#define kMinFlagLong "-scaleMin"
#define kMaxFlag "-smx" 
#define kMaxFlagLong "-scaleMax"
#define kRotateNoiseFlag "-rno" 
#define kRotateNoiseFlagLong "-rotateNoise"
#define kWeightFlag "-bwt" 
#define kWeightFlagLong "-brushWeight"
#define kNormalFlag "-anl" 
#define kNormalFlagLong "-alongNormal"
#define kWriteCacheFlag "-wch" 
#define kWriteCacheFlagLong "-writeCache"
#define kReadCacheFlag "-rch" 
#define kReadCacheFlagLong "-readCache"
#define kMultiCreateFlag "-mct" 
#define kMultiCreateFlagLong "-multiCreate"
#define kInstanceGroupCountFlag "-igc" 
#define kInstanceGroupCountFlagLong "-instanceGroupCount"
#define kPlantTypeFlag "-pty"
#define kPlantTypeFlagLong "-plantType"
#define kStickToGroundFlag "-stg"
#define kStickToGroundFlagLong "-stickToGround"
#define kSelectVizFlag "-slv"
#define kSelectVizFlagLong "-selectViz"
#define kNoiseFreqFlag "-nfr"
#define kNoiseFreqFlagLong "-noiseFrequency"
#define kNoiseLacunarityFlag "-nlc"
#define kNoiseLacunarityFlagLong "-noiseLacunarity"
#define kNoiseOctaveFlag "-nov"
#define kNoiseOctaveFlagLong "-noiseOctave"
#define kNoiseLevelFlag "-nlv"
#define kNoiseLevelFlagLong "-noiseLevel"
#define kNoiseGainFlag "-ngn"
#define kNoiseGainFlagLong "-noiseGain"
#define kNoiseOriginXFlag "-nox"
#define kNoiseOriginXFlagLong "-noiseOriginX"
#define kNoiseOriginYFlag "-noy"
#define kNoiseOriginYFlagLong "-noiseOriginY"
#define kNoiseOriginZFlag "-noz"
#define kNoiseOriginZFlagLong "-noiseOriginZ"
#define kNoiseOriginVecFlag "-ogv"
#define kNoiseOriginVecFlagLong "-noiseOriginVec"
#define kImageSamplerFlag "-msp"
#define kImageSamplerFlagLong "-imageSampler"
#define kReshuffleSampleFlag "-rss"
#define kReshuffleSampleFlagLong "-reshuffleSample"
#define kFilterPortionFlag "-fpt"
#define kFilterPortionFlagLong "-filterPortion"
#define kSampleCountFlag "-spc"
#define kSampleCountFlagLong "-sampleCount"
#define kViewSelectSampleFlag "-vsp"
#define kViewSelectSampleFlagLong "-viewSelectSample"
#define kVizViewGridFlag "-vvg"
#define kVizViewGridFlagLong "-vizViewGrid"
#define kVizStatFlag "-vst"
#define kVizStatFlagLong "-vizStat"
#define kVizEditGroundFlag "-veg"
#define kVizEditGroundFlagLong "-vizEditGround"

proxyPaintContextCmd::proxyPaintContextCmd() {}

MPxContext* proxyPaintContextCmd::makeObj()
{
	fContext = new proxyPaintContext();
	return fContext;
}

void* proxyPaintContextCmd::creator()
{
	return new proxyPaintContextCmd;
}

MStatus proxyPaintContextCmd::doEditFlags()
{
	MStatus status = MS::kSuccess;
	
	MArgParser argData = parser();
	
	if (argData.isFlagSet(kOptFlag)) 
	{
		unsigned mode;
		status = argData.getFlagArgument(kOptFlag, 0, mode);
		if (!status) {
			status.perror("mode flag parsing failed.");
			return status;
		}
		fContext->setOperation(mode);
	}
	
	if (argData.isFlagSet(kLsegFlag)) 
	{
		double lseg;
		status = argData.getFlagArgument(kLsegFlag, 0, lseg);
		if (!status) {
			status.perror("lseg flag parsing failed.");
			return status;
		}
		fContext->setBrushRadius(lseg);
	}
	
	if (argData.isFlagSet(kWeightFlag)) 
	{
		double wei;
		status = argData.getFlagArgument(kWeightFlag, 0, wei);
		if (!status) {
			status.perror("lseg flag parsing failed.");
			return status;
		}
		fContext->setBrushWeight(wei);
	}
	
	if (argData.isFlagSet(kMinFlag)) 
	{
		double noi;
		status = argData.getFlagArgument(kMinFlag, 0, noi);
		if (!status) {
			status.perror("scale min flag parsing failed.");
			return status;
		}
		fContext->setScaleMin(noi);
	}
	
	if (argData.isFlagSet(kMaxFlag)) 
	{
		double noi;
		status = argData.getFlagArgument(kMaxFlag, 0, noi);
		if (!status) {
			status.perror("scale max flag parsing failed.");
			return status;
		}
		fContext->setScaleMax(noi);
	}
	
	if (argData.isFlagSet(kRotateNoiseFlag)) 
	{
		double noi;
		status = argData.getFlagArgument(kRotateNoiseFlag, 0, noi);
		if (!status) {
			status.perror("rotate noise flag parsing failed.");
			return status;
		}
		fContext->setRotationNoise(noi);
	}
	
	if (argData.isFlagSet(kNormalFlag)) 
	{
		unsigned aln;
		status = argData.getFlagArgument(kNormalFlag, 0, aln);
		if (!status) {
			status.perror("normal flag parsing failed.");
			return status;
		}
		fContext->setGrowAlongNormal(aln);
	}
	
	if (argData.isFlagSet(kWriteCacheFlag)) 
	{
		MString ch;
		status = argData.getFlagArgument(kWriteCacheFlag, 0, ch);
		if (!status) {
			status.perror("cache out flag parsing failed.");
			return status;
		}
		fContext->setWriteCache(ch);
	}
	
	if (argData.isFlagSet(kReadCacheFlag)) 
	{
		MString ch;
		status = argData.getFlagArgument(kReadCacheFlag, 0, ch);
		if (!status) {
			status.perror("cache in flag parsing failed.");
			return status;
		}
		fContext->setReadCache(ch);
	}
	
	if (argData.isFlagSet(kMultiCreateFlag)) {
		unsigned mcr;
		status = argData.getFlagArgument(kMultiCreateFlag, 0, mcr);
		if (!status) {
			status.perror("multi create flag parsing failed.");
			return status;
		}
		fContext->setMultiCreate(mcr);
	}
	
	if (argData.isFlagSet(kInstanceGroupCountFlag)) {
		unsigned igc;
		status = argData.getFlagArgument(kInstanceGroupCountFlag, 0, igc);
		if (!status) {
			status.perror("instance group count flag parsing failed.");
			return status;
		}
		fContext->setInstanceGroupCount(igc);
	}
	
	if (argData.isFlagSet(kMinMarginFlag)) {
		double margin = 1.0;
		if (argData.getFlagArgument(kMinMarginFlag, 0, margin) )
			fContext->setMinCreateMargin(margin);
	}
    
    if (argData.isFlagSet(kMaxMarginFlag)) {
		double margin = 1.0;
		if (argData.getFlagArgument(kMaxMarginFlag, 0, margin) )
			fContext->setMaxCreateMargin(margin);
	}
	
	if (argData.isFlagSet(kPlantTypeFlag)) {
		int pty = 0;
		if (argData.getFlagArgument(kPlantTypeFlag, 0, pty) )
			fContext->setPlantType(pty);
	}
	
	if (argData.isFlagSet(kStickToGroundFlag)) {
		int stg = 1;
		if (argData.getFlagArgument(kStickToGroundFlag, 0, stg) )
			fContext->setStickToGround(stg>0);
	}
	
	if(argData.isFlagSet(kSelectVizFlag)) {
		fContext->selectViz();
	}
	
	if (argData.isFlagSet(kNoiseFreqFlag)) {
		double freq = 1;
		if (argData.getFlagArgument(kNoiseFreqFlag, 0, freq) )
			fContext->setNoiseFrequency(freq);
	}
	
	if (argData.isFlagSet(kNoiseLacunarityFlag)) {
		double lac = 1;
		if (argData.getFlagArgument(kNoiseLacunarityFlag, 0, lac) )
			fContext->setNoiseLacunarity(lac);
	}
	
	if (argData.isFlagSet(kNoiseOctaveFlag)) {
		int oct = 1;
		if (argData.getFlagArgument(kNoiseOctaveFlag, 0, oct) )
			fContext->setNoiseOctave(oct);
	}
	
	if (argData.isFlagSet(kNoiseLevelFlag)) {
		double lev = 1;
		if (argData.getFlagArgument(kNoiseLevelFlag, 0, lev) )
			fContext->setNoiseLevel(lev);
	}
	
	if (argData.isFlagSet(kNoiseGainFlag)) {
		double gan = 1;
		if (argData.getFlagArgument(kNoiseGainFlag, 0, gan) )
			fContext->setNoiseGain(gan);
	}
	
	if (argData.isFlagSet(kNoiseOriginXFlag)) {
		double nox = 1;
		if (argData.getFlagArgument(kNoiseOriginXFlag, 0, nox) )
			fContext->setNoiseOriginX(nox);
	}
	
	if (argData.isFlagSet(kNoiseOriginYFlag)) {
		double noy = 1;
		if (argData.getFlagArgument(kNoiseOriginYFlag, 0, noy) )
			fContext->setNoiseOriginY(noy);
	}
	
	if (argData.isFlagSet(kNoiseOriginZFlag)) {
		double noz = 1;
		if (argData.getFlagArgument(kNoiseOriginZFlag, 0, noz) )
			fContext->setNoiseOriginZ(noz);
	}
	
	if (argData.isFlagSet(kNoiseOriginVecFlag)) {
		double originx = 1.0;
		double originy = 1.0;
		double originz = 1.0;
		argData.getFlagArgument(kNoiseOriginVecFlag, 0, originx);
		argData.getFlagArgument(kNoiseOriginVecFlag, 1, originy);
		argData.getFlagArgument(kNoiseOriginVecFlag, 2, originz);
		
		fContext->setNoiseOrigin(originx, originy, originz);
	}
	
	if (argData.isFlagSet(kImageSamplerFlag)) {
		MString imageName;
		if (argData.getFlagArgument(kImageSamplerFlag, 0, imageName) ) {
			fContext->setImageSamplerName(imageName);
		}
	}
	
	if (argData.isFlagSet(kReshuffleSampleFlag)) {
		fContext->reshuffleSamples();
	}
	
	if (argData.isFlagSet(kViewSelectSampleFlag)) {
		fContext->viewDependentSelectSamples();
	}
	
	if (argData.isFlagSet(kFilterPortionFlag)) {
		double portion = 1.0;
		if (argData.getFlagArgument(kFilterPortionFlag, 0, portion) )
			fContext->setFilterPortion(portion);
	}
	
	if (argData.isFlagSet(kVizViewGridFlag)) {
		int vvg = 0;
		if (argData.getFlagArgument(kVizViewGridFlag, 0, vvg) )
			fContext->setShowVizGrid(vvg);
	}
	
	if (argData.isFlagSet(kVizEditGroundFlag)) {
		int veg = 0;
		if (argData.getFlagArgument(kVizEditGroundFlag, 0, veg) )
			fContext->setEditVizGround(veg);
	}
	
	return MS::kSuccess;
}

MStatus proxyPaintContextCmd::doQueryFlags()
{
	MArgParser argData = parser();
	
	if (argData.isFlagSet(kOptFlag)) {
		setResult((int)fContext->getOperation());
	}
	
	if (argData.isFlagSet(kLsegFlag)) {
		setResult((float)fContext->getBrushRadius());
	}
	
	if (argData.isFlagSet(kWeightFlag)) {
		setResult((float)fContext->getBrushWeight());
	}
	
	if (argData.isFlagSet(kMinFlag)) {
		setResult((float)fContext->getScaleMin());
	}
	
	if (argData.isFlagSet(kMaxFlag)) {
		setResult((float)fContext->getScaleMax());
	}
	
	if (argData.isFlagSet(kRotateNoiseFlag)) {
		setResult((float)fContext->getRotationNoise());
	}
	
	if (argData.isFlagSet(kNormalFlag)) {
		setResult((int)fContext->getGrowAlongNormal());
	}
	
	if (argData.isFlagSet(kMultiCreateFlag)) {
		setResult((int)fContext->getMultiCreate());
	}
	
	if (argData.isFlagSet(kInstanceGroupCountFlag)) {
		setResult((int)fContext->getInstanceGroupCount());
	}
	
	if (argData.isFlagSet(kMinMarginFlag))
		setResult((float)fContext->minCreateMargin() );
        
    if (argData.isFlagSet(kMaxMarginFlag))
		setResult((float)fContext->maxCreateMargin() );
	
	if (argData.isFlagSet(kPlantTypeFlag))
		setResult(fContext->plantType() );
		
	if (argData.isFlagSet(kStickToGroundFlag))
		setResult(fContext->stickToGround() );
		
	if (argData.isFlagSet(kNoiseFreqFlag))
		setResult(fContext->noiseFrequency() );
		
	if (argData.isFlagSet(kNoiseLacunarityFlag))
		setResult(fContext->noiseLacunarity() );
		
	if (argData.isFlagSet(kNoiseOctaveFlag))
		setResult(fContext->noiseOctave() );
	
	if (argData.isFlagSet(kNoiseLevelFlag))
		setResult(fContext->noiseLevel() );
		
	if (argData.isFlagSet(kNoiseGainFlag))
		setResult(fContext->noiseGain() );
		
	if (argData.isFlagSet(kNoiseOriginXFlag))
		setResult(fContext->noiseOriginX() );
		
	if (argData.isFlagSet(kNoiseOriginYFlag))
		setResult(fContext->noiseOriginY() );
		
	if (argData.isFlagSet(kNoiseOriginZFlag)) {
		setResult(fContext->noiseOriginZ() );
	}
	
	if (argData.isFlagSet(kImageSamplerFlag)) {
		setResult(fContext->imageSamplerName() );
	}
	
	if (argData.isFlagSet(kFilterPortionFlag)) {
		setResult(fContext->filterPortion() );
	}
	
	if (argData.isFlagSet(kSampleCountFlag)) {
		setResult(fContext->numVisibleSamples() );
	}
	
	if (argData.isFlagSet(kVizViewGridFlag)) {
		setResult(fContext->getShowVizGrid() );
	}
	
	if (argData.isFlagSet(kVizEditGroundFlag)) {
		setResult(fContext->getEditVizGround() );
	}
	
	if (argData.isFlagSet(kVizStatFlag)) {
		setVizStatResult();
	}
	
	return MS::kSuccess;
}

MStatus proxyPaintContextCmd::appendSyntax()
{
	MSyntax mySyntax = syntax();
	
	MStatus stat;
	stat = mySyntax.addFlag(kOptFlag, kOptFlagLong, MSyntax::kUnsigned);
	if(!stat)
	{
		MGlobal::displayInfo("failed to add option arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kLsegFlag, kLsegFlagLong, MSyntax::kDouble);
	if(!stat)
	{
		MGlobal::displayInfo("failed to add radius arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kWeightFlag, kWeightFlagLong, MSyntax::kDouble);
	if(!stat)
	{
		MGlobal::displayInfo("failed to add weight arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kMinFlag, kMinFlagLong, MSyntax::kDouble);
	if(!stat)
	{
		MGlobal::displayInfo("failed to add min arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kMaxFlag, kMaxFlagLong, MSyntax::kDouble);
	if(!stat)
	{
		MGlobal::displayInfo("failed to add max arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kNormalFlag, kNormalFlagLong, MSyntax::kUnsigned);
	if(!stat)
	{
		MGlobal::displayInfo("failed to add normal arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kWriteCacheFlag, kWriteCacheFlagLong, MSyntax::kString);
	if(!stat)
	{
		MGlobal::displayInfo("failed to add cache out arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kReadCacheFlag, kReadCacheFlagLong, MSyntax::kString);
	if(!stat)
	{
		MGlobal::displayInfo("failed to add cache in arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kRotateNoiseFlag, kRotateNoiseFlagLong, MSyntax::kDouble);
	if(!stat)
	{
		MGlobal::displayInfo("failed to add rotate noise arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kMultiCreateFlag, kMultiCreateFlagLong, MSyntax::kUnsigned);
	if(!stat) {
		MGlobal::displayInfo("failed to add multi create arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kInstanceGroupCountFlag, kInstanceGroupCountFlagLong, MSyntax::kUnsigned);
	if(!stat) {
		MGlobal::displayInfo("failed to add instance group count arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kMinMarginFlag, kMinMarginFlagLong, MSyntax::kDouble);
	if(!stat) {
		MGlobal::displayInfo("failed to add min margin arg");
		return MS::kFailure;
	}
    
    stat = mySyntax.addFlag(kMaxMarginFlag, kMaxMarginFlagLong, MSyntax::kDouble);
	if(!stat) {
		MGlobal::displayInfo("failed to add max margin arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kPlantTypeFlag, kPlantTypeFlagLong, MSyntax::kLong);
	if(!stat) {
		MGlobal::displayInfo("failed to add plantType arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kStickToGroundFlag, kStickToGroundFlagLong, MSyntax::kLong);
	if(!stat) {
		MGlobal::displayInfo("failed to add stickToGround arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kSelectVizFlag, kSelectVizFlagLong, MSyntax::kNoArg);
	if(!stat) {
		MGlobal::displayInfo("failed to add selectViz arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kNoiseFreqFlag, kNoiseFreqFlagLong, MSyntax::kDouble);
	if(!stat) {
		MGlobal::displayInfo("failed to add noise frequency arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kNoiseLacunarityFlag, kNoiseLacunarityFlagLong, MSyntax::kDouble);
	if(!stat) {
		MGlobal::displayInfo("failed to add noise lacunarity arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kNoiseOctaveFlag, kNoiseOctaveFlagLong, MSyntax::kLong);
	if(!stat) {
		MGlobal::displayInfo("failed to add noise octave arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kNoiseLevelFlag, kNoiseLevelFlagLong, MSyntax::kDouble);
	if(!stat) {
		MGlobal::displayInfo("failed to add noise level arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kNoiseGainFlag, kNoiseGainFlagLong, MSyntax::kDouble);
	if(!stat) {
		MGlobal::displayInfo("failed to add noise gain arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kNoiseOriginXFlag, kNoiseOriginXFlagLong, MSyntax::kDouble);
	if(!stat) {
		MGlobal::displayInfo("failed to add noise origin x arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kNoiseOriginYFlag, kNoiseOriginYFlagLong, MSyntax::kDouble);
	if(!stat) {
		MGlobal::displayInfo("failed to add noise origin y arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kNoiseOriginZFlag, kNoiseOriginZFlagLong, MSyntax::kDouble);
	if(!stat) {
		MGlobal::displayInfo("failed to add noise origin z arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kImageSamplerFlag, kImageSamplerFlagLong, MSyntax::kString);
	if(!stat) {
		MGlobal::displayInfo("failed to add image sampler arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kReshuffleSampleFlag, kReshuffleSampleFlagLong, MSyntax::kNoArg);
	if(!stat) {
		MGlobal::displayInfo("failed to add rebuild sample arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kFilterPortionFlag, kFilterPortionFlagLong, MSyntax::kDouble);
	if(!stat) {
		MGlobal::displayInfo("failed to add filter portion arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kNoiseOriginVecFlag, kNoiseOriginVecFlagLong, MSyntax::kDouble, MSyntax::kDouble, MSyntax::kDouble);
	if(!stat) {
		MGlobal::displayInfo("failed to add noise origin vec arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kSampleCountFlag, kSampleCountFlagLong, MSyntax::kNoArg);
	if(!stat) {
		MGlobal::displayInfo("failed to add sample count arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kViewSelectSampleFlag, kViewSelectSampleFlagLong, MSyntax::kNoArg);
	if(!stat) {
		MGlobal::displayInfo("failed to view select sample arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kVizViewGridFlag, kVizViewGridFlagLong, MSyntax::kLong);
	if(!stat) {
		MGlobal::displayInfo("failed to viz view grid arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kVizStatFlag, kVizStatFlagLong, MSyntax::kNoArg);
	if(!stat) {
		MGlobal::displayInfo("failed to viz stat arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kVizEditGroundFlag, kVizEditGroundFlagLong, MSyntax::kLong);
	if(!stat) {
		MGlobal::displayInfo("failed to viz edit ground arg");
		return MS::kFailure;
	}
	
	return stat;
}

void proxyPaintContextCmd::setVizStatResult()
{
	std::map<std::string, std::string > stats;
	fContext->getVizStatistics(stats);
	MString res;
	std::map<std::string, std::string >::iterator it = stats.begin();
	for(;it != stats.end();++it) {
		res += MString(it->first.c_str() );
		res += MString("|");
		res += MString(it->second.c_str() );
		res += MString("|");
	}
	
	setResult(res);
}
