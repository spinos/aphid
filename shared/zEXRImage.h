#ifndef ZFN_EXR_H
#define ZFN_EXR_H
#include <BaseImage.h>
#include <vector>
#include <ImfRgbaFile.h>
#include <ImfHeader.h>
#include <ImfOutputFile.h>
#include <ImfInputFile.h>
#include <ImfChannelList.h>
#include <ImfArray.h>

#include <ImfBoxAttribute.h>
#include <ImfChannelListAttribute.h>
#include <ImfCompressionAttribute.h>
#include <ImfChromaticitiesAttribute.h>
#include <ImfFloatAttribute.h>
#include <ImfEnvmapAttribute.h>
#include <ImfDoubleAttribute.h>
#include <ImfIntAttribute.h>
#include <ImfLineOrderAttribute.h>
#include <ImfMatrixAttribute.h>
#include <ImfOpaqueAttribute.h>
#include <ImfStringAttribute.h>
#include <ImfVecAttribute.h>

class ZEXRImage;

class ZEXRSampler
{
public:
	ZEXRSampler();
	~ZEXRSampler();
	void setPixels(ZEXRImage * src);
	void reduceFrom(ZEXRSampler * src);
	void setWidth(int w);
	half* getPixels() const;
	int getWidth() const;
	void sample(float u, float v, int count, float * dst) const;
	half *_pixels;
	int _width, _rank;
};

class ZEXRImage : public BaseImage
{
public:
	ZEXRImage();
	ZEXRImage(const char* filename, bool loadMipmap = false);
	~ZEXRImage(void);
	
	virtual bool doRead(const std::string & filename);
	virtual void doClear();
	virtual const char * formatName() const;
	
	virtual void allWhite();
	virtual void allBlack();
	
	virtual void sample(float u, float v, int count, float * dst) const;
	
	void sample(float u, float v, float level, int count, float * dst) const;
	
	int getNumMipmaps() const;
	
	virtual void applyMask(BaseImage * another);
	
	static bool isAnOpenExrFile(const std::string& filename);
	static void listExrChannelNames(const std::string& filename, std::vector<std::string>& dst);
	
	half *_pixels;
	float * m_zData;
	
	std::vector<ZEXRSampler*>_mipmaps;
	int _numMipmaps;
	
private:
	void readPixels(Imf::InputFile& file);
	void readZ(Imf::InputFile& file);
	void setupMipmaps();
	bool findZChannel(Imf::InputFile & file);
private:
    std::string m_zChannelName;
    bool m_hasMipmap;
};
#endif
