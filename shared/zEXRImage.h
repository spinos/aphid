#ifndef ZFN_EXR_H
#define ZFN_EXR_H
#include <BaseImage.h>
#include <vector>
#include <OpenEXR/half.h>
#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfHeader.h>

class ZEXRImage;
using namespace IMATH_NAMESPACE;
using namespace OPENEXR_IMF_NAMESPACE;
class ZEXRSampler
{
public:
	ZEXRSampler();
	virtual ~ZEXRSampler();
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
	ZEXRImage(const char * filename);
	ZEXRImage(bool loadMipmap);
	virtual ~ZEXRImage();
	
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
	static void PrintChannelNames(const std::string& filename);
	
	bool getTile1(float * dst, int ind, int tileSize, int rank = 3) const;
	bool getTile(float * dst, int x, int y, int tileSize, int rank = 3) const;
	half *_pixels;
	float * m_zData;
	
	std::vector<ZEXRSampler*>_mipmaps;
	int _numMipmaps;
	
private:
    ChannelRank checkColors(const Header & head) const;
	void readPixels(InputFile& file);
	void readZ(InputFile& file);
	void setupMipmaps();
	bool findZChannel(const ChannelList &channels);
private:
    std::string m_zChannelName;
    bool m_hasMipmap;
};
#endif
