#ifndef APHID_EXR_IMAGE_H
#define APHID_EXR_IMAGE_H

#include "BaseImage.h"

namespace aphid {
class ExrImage : public BaseImage
{
    char *_pixels;
    
public:
	ExrImage();
	virtual ~ExrImage();
	
	virtual IFormat formatName() const;
	
	bool getTile(float * dst, const int ind, int tileSize, int rank = 3) const;
	
	static bool IsOpenExrFile(const std::string& filename);

	virtual void sample(float u, float v, int count, float * dst) const;
	virtual void sampleRed(float * y) const;
	virtual void resampleRed(float * y, int sx, int sy) const;
	virtual void sampleRed(Array3<float> & y) const;
	
protected:
	virtual bool readImage(const std::string & filename);
		
private:
    void tileCoord(const int ind, const int tileSize, int & x, int & y) const;
	void clear();
	void boxSample(const float & u, const float & v, const int & count, 
			const int & w, const int & h, const int & colorRank,
			float * dst) const;
	void sample(float & dst,
			const int & u, const int & v,
			const int & w, const int & h, const int & colorRank,
			const int & offset) const;
	
};
}
#endif        //  #ifndef APHID_EXR_IMAGE_H

