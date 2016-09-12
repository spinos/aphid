#ifndef LFPARAMETER_H
#define LFPARAMETER_H

#include <string>
#include <vector>
#include <ExrImage.h>
#include <ATypes.h>

namespace lfr {

class LfParameter {

    struct ImageInd {
		aphid::ExrImage * _image;
		int _ind;
	};
	
#define MAX_NUM_OPENED_IMAGES 16 
	ImageInd m_openedImages[MAX_NUM_OPENED_IMAGES];
	int m_currentImage;
	
	std::vector<std::string > m_imageNames;
	std::vector<int > m_numPatches;
	std::vector<Int2 > m_imageSizes;
	std::string m_searchPath;
	
	int m_nthread;
	int m_maxIter;
/// n x n atom
	int m_atomSize;
/// d/m where m = n^2 * 3 
	float m_overcomplete;
/// total num patches
	int m_numTotalPatches;
/// dictionary image size
	int m_dictWidth, m_dictHeight;
	bool m_isValid, m_isHelp;
public:
	LfParameter(int argc, char *argv[]);
	virtual ~LfParameter();
	
	const bool & isValid() const;
	const bool & isHelp() const;
	int numThread() const;
	const int & atomSize() const;
	int dictionaryLength() const;
	int totalNumPatches() const;
	int totalNumPixels() const;
	int dimensionOfX() const;
	int randomImageInd() const;
	int numImages() const;
	std::string imageName(int i) const;
	int imageNumPatches(int i) const;
	int imageNumPixels(int i) const;
	int maxIterations() const;
	
	bool isImageOpened(const int ind, int & idx) const;
	aphid::ExrImage *openImage(const int ind);
	
	void getDictionaryImageSize(int & x, int & y) const;
	void getImageSize(int & x, int & y, const int & i) const;
	
	void printHelp() const;
	
protected:
	virtual void printVersion() const;
	virtual void printUsage() const;
	virtual void printDescription() const;
	virtual void printOptions() const;
	
private:
	bool searchImagesIn(const char * dirname);
	bool countPatches();
	
};

}
#endif        //  #ifndef LFPARAMETER_H

