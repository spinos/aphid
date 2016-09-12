#include "LfParameter.h"

#include <boost/lexical_cast.hpp>
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/filesystem/convenience.hpp"
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/case_conv.hpp>

#define _WIN32
#include <iostream>

using namespace aphid;

namespace lfr {

LfParameter::LfParameter(int argc, char *argv[])
{
	int i = 0;
	for(;i<MAX_NUM_OPENED_IMAGES;++i) {
		m_openedImages[i]._ind = -1;
		m_openedImages[i]._image = NULL;
	}
	m_currentImage = 0;
	m_isValid = false;
	m_isHelp = false;
	m_atomSize = 8;
	m_overcomplete = 1.f;
	m_nthread = 2;
	m_maxIter = 1000;
	m_searchPath = "";
	m_dictWidth = m_dictHeight = 0;
	bool foundImages = false;
	if(argc == 1) {
		m_isValid = searchImagesIn("./");
	}
	else {
		i = 1;
		for(;i<argc;++i) {
			if(strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
				m_isHelp = true;
				break;
			}
			if(strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--thread") == 0) {
				if(i==argc-1) {
					std::cout<<"\n --thread value is not set";
					break;
				}
				try {
					m_nthread = boost::lexical_cast<int>(argv[i+1]);
					if(m_nthread < 1) {
						std::cout<<"\n bad --thread value (< 1)";
						break;
					}
				}
				catch(const boost::bad_lexical_cast &) {
					std::cout<<"\n bad --thread value "<<argv[i+1];
					break;
				}
			}
			if(strcmp(argv[i], "-mi") == 0 || strcmp(argv[i], "--maxIteration") == 0) {
				if(i==argc-1) {
					std::cout<<"\n --maxIteration value is not set";
					break;
				}
				try {
					m_maxIter = boost::lexical_cast<int>(argv[i+1]);
					if(m_maxIter < 1) {
						std::cout<<"\n bad --maxIteration value (< 1)";
						break;
					}
				}
				catch(const boost::bad_lexical_cast &) {
					std::cout<<"\n bad --maxIteration value "<<argv[i+1];
					break;
				}
			}
			if(strcmp(argv[i], "-as") == 0 || strcmp(argv[i], "--atomSize") == 0) {
				if(i==argc-1) {
					std::cout<<"\n --atomSize value is not set";
					break;
				}
				try {
					m_atomSize = boost::lexical_cast<int>(argv[i+1]);
					if(m_atomSize < 8) {
						std::cout<<"\n bad --atomSize value (< 8)";
						break;
					}
				}
				catch(const boost::bad_lexical_cast &) {
					std::cout<<"\n bad --atomSize value "<<argv[i+1];
					break;
				}
			}
			if(strcmp(argv[i], "-oc") == 0 || strcmp(argv[i], "--overcomplete") == 0) {
				if(i==argc-1) {
					std::cout<<"\n --overcomplete value is not set";
					break;
				}
				try {
					m_overcomplete = boost::lexical_cast<float>(argv[i+1]);
					if(m_overcomplete < 1.f) {
						std::cout<<"\n bad --m_overcomplete value (< 1.0)";
						break;
					}
				}
				catch(const boost::bad_lexical_cast &) {
					std::cout<<"\n bad --m_overcomplete value "<<argv[i+1];
					break;
				}
			}
			if(i==argc-1) {
				if(boost::filesystem::exists(argv[i])) {
					boost::filesystem::path p(argv[i]);
					if(boost::filesystem::is_regular_file(p) ) {
						m_imageNames.push_back(argv[i]);
						std::cout<<"\n found image "<<argv[i];
						m_isValid = true;
					}
					else {
						m_searchPath = argv[i];
						m_isValid = searchImagesIn(argv[i]);
					}
				}
			}
		}
	}
	if(m_isValid) {
		std::cout<<"\n image atom size "<<m_atomSize;
		std::cout<<"\n dictionary overcompleteness "<<m_overcomplete;
		m_isValid = countPatches();
	}
	if(m_isValid) {
		const float p = dictionaryLength();
		int w = sqrt(p); 
		if(w*w < p) w++;
		int h = p / w;
		if(h*w < p) h++;
		m_dictWidth = w * m_atomSize;
		m_dictHeight = h * m_atomSize;
		std::cout<<"\n dictionary image size ("<<m_dictWidth<<", "<<m_dictHeight<<")"
		<<"\n passed parameter check\n";
	}
}

LfParameter::~LfParameter() {}

const bool & LfParameter::isValid() const
{ return m_isValid; }

const bool & LfParameter::isHelp() const
{ return m_isHelp; }

bool LfParameter::searchImagesIn(const char * dirname)
{
	m_imageNames.clear();
	if(!boost::filesystem::exists(dirname)) {
		std::cout<<"\n dir doesn't exist "<<dirname;
		return false;
	}
	std::cout<<"\n searching images in dir "<<dirname;
	
	const std::string ext(".exr");
	boost::filesystem::path head_path(dirname);
	boost::filesystem::directory_iterator end_iter;
	for ( boost::filesystem::directory_iterator itdir( head_path );
		  itdir != end_iter;
		  ++itdir ) {
		if ( boost::filesystem::is_regular_file( itdir->status() ) ) {
			const std::string fn = itdir->path().filename().string();
			if(!boost::algorithm::starts_with(fn, ".")) {
			    
				std::string fileext = boost::filesystem::extension(itdir->path().string());
				boost::algorithm::to_lower(fileext);
				if(fileext == ext) {
					m_imageNames.push_back( boost::filesystem::basename(itdir->path()) + ".exr");
					std::cout<<"\n "<<m_imageNames.back();
				}
			}
		}
	}
	
	std::cout<<"\n  found "<<m_imageNames.size();
	return true;
}

bool LfParameter::countPatches()
{
	m_numPatches.clear();
	m_imageSizes.clear();
	
    ExrImage img;
	m_numTotalPatches = 0;
	const int n = numImages();
	for(int i=0; i<n;++i) {
		if(img.read(imageName(i) ) ) {
			m_numPatches.push_back( (img.getWidth() / m_atomSize) * (img.getHeight() / m_atomSize) );
			m_imageSizes.push_back(Int2(img.getWidth() / m_atomSize * m_atomSize, img.getHeight()  / m_atomSize * m_atomSize ) );
			m_numTotalPatches += m_numPatches.back();
        }
	}
	std::cout<<"\n num total patches "<<m_numTotalPatches;
	return m_numTotalPatches > 0;
}

const int & LfParameter::atomSize() const
{ return m_atomSize; }

int LfParameter::dictionaryLength() const
{ return dimensionOfX() * m_overcomplete; }

std::string LfParameter::imageName(int i) const
{ 
	if(m_searchPath.size() < 2)
		return m_imageNames[i];
		
	return m_searchPath + "/" + m_imageNames[i]; 
}

int LfParameter::imageNumPatches(int i) const
{ return m_numPatches[i]; }

int LfParameter::imageNumPixels(int i) const
{ return imageNumPatches(i) * m_atomSize * m_atomSize; }

int LfParameter::totalNumPatches() const
{ return m_numTotalPatches; }

int LfParameter::totalNumPixels() const
{ return m_numTotalPatches * m_atomSize * m_atomSize; }

int LfParameter::dimensionOfX() const
{ return m_atomSize * m_atomSize * 3; }

int LfParameter::randomImageInd() const
{ 
	if(m_imageNames.size() < 2) return 0; 
	return rand() % m_imageNames.size();
}

int LfParameter::numImages() const
{ return m_imageNames.size(); }

bool LfParameter::isImageOpened(const int ind, int & idx) const
{
	int i = 0;
	for(;i<MAX_NUM_OPENED_IMAGES;++i) {
		if(m_openedImages[i]._ind < 0) break;
		
		if( m_openedImages[i]._ind == ind ) {
		    idx = i;
			// std::cout<<" "<<i<<" is opened "<<m_openedImages[i]._image;
			return true;
		}
	}
	return false;
}

ExrImage *LfParameter::openImage(const int ind)
{
    int idx;
    if(isImageOpened(ind, idx)) {
        return m_openedImages[idx]._image;
    }
    
    idx = m_currentImage;
	if(m_openedImages[idx]._ind < 0) {
		m_openedImages[idx]._image = new ExrImage;
	}
	m_openedImages[idx]._ind = ind;
	m_openedImages[idx]._image->read(imageName(ind));
	//std::cout<<" open "<<m_openedImages[idx]._image<<"   ";
	m_currentImage = (m_currentImage + 1) % MAX_NUM_OPENED_IMAGES;
	return m_openedImages[idx]._image;
}

void LfParameter::getDictionaryImageSize(int & x, int & y) const
{ x = m_dictWidth; y = m_dictHeight; }

void LfParameter::getImageSize(int & x, int & y, const int & i) const
{ 
	x = m_imageSizes[i].x; 
	y = m_imageSizes[i].y; 
}

int LfParameter::numThread() const
{ return m_nthread; }

int LfParameter::maxIterations() const
{ return m_maxIter; }

void LfParameter::printHelp() const
{
	printVersion();
	printDescription();
	printUsage();
	printOptions();
}

void LfParameter::printVersion() const
{ std::cout<<"\n lfr (Light Field Research) version 20151122"; }

void LfParameter::printDescription() const
{
	std::cout<<"\nDescription:\n lfr learns the underlying pattern of input images.";
}

void LfParameter::printUsage() const
{
	std::cout<<"\nUsage:\n lfr [option] [file]"
	<<"\n Input file must be image of OpenEXR format. Last input can be a directory"
	<<"\n to seach for any file with name ending in .exr." 
	<<"\n If no file or directory is provided, current dir will be searched.";
}

void LfParameter::printOptions() const	
{
	std::cout<<"\nOptions:\n -as or --atomSize    integer    size of image atoms, no less than 8, default is 8"
	<<"\n -t or --thread    integer    number of threads to use, limit to 1 - 24, default is 2"
	<<"\n -mi or --maxIteration    integer    limit of iterations, default is 1000"
	<<"\n -oc or --overcomplete    float    overcompleteness of dictionary, d/m, no less than 1.0, default is 1.0"
	<<"\n -h or --help    print this information"
	<<"\n";
}

}
