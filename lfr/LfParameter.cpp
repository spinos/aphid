#include "LfParameter.h"

#include <boost/lexical_cast.hpp>
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/filesystem/convenience.hpp"
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/case_conv.hpp>

#define _WIN32
#include <zEXRImage.h>
#include <OpenEXR/ImathLimits.h>

namespace lfr {

LfParameter::LfParameter(int argc, char *argv[])
{
	std::cout<<"\n exr limit "<<Imath::limits<int>::min();
	std::cout<<"\n test min "<<std::min<int>(-99, -98);
	std::cout<<"\n test abs "<<std::abs<int>(-75);
	std::cout<<"\n lfr (Light Field Research) version 20151127";
	int i = 0;
	for(;i<MAX_NUM_OPENED_IMAGES;++i) {
		m_openedImages[i]._ind = -1;
		m_openedImages[i]._image = NULL;
	}
	m_currentImage = 0;
	m_isValid = false;
	m_atomSize = 10;
	m_dictionaryLength = 1024;
	bool foundImages = false;
	if(argc == 1) {
		m_isValid = searchImagesIn("./");
	}
	else {
		i = 1;
		for(;i<argc;++i) {
			if(strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
				PrintHelp();
			}
			if(strcmp(argv[i], "-as") == 0 || strcmp(argv[i], "--atomSize") == 0) {
				if(i==argc-1) {
					std::cout<<"\n --atomSize value is not set";
					break;
				}
				try {
					m_atomSize = boost::lexical_cast<int>(argv[i+1]);
					if(m_atomSize < 10) {
						std::cout<<"\n bad --atomSize value (< 10)";
						break;
					}
				}
				catch(const boost::bad_lexical_cast &) {
					std::cout<<"\n bad --atomSize value "<<argv[i+1];
					break;
				}
			}
			if(strcmp(argv[i], "-dl") == 0 || strcmp(argv[i], "--dictionaryLength") == 0) {
				if(i==argc-1) {
					std::cout<<"\n --dictionaryLength value is not set";
					break;
				}
				try {
					m_dictionaryLength = boost::lexical_cast<int>(argv[i+1]);
					if(m_dictionaryLength < 1024) {
						std::cout<<"\n bad --dictionaryLength value (< 1024)";
						break;
					}
				}
				catch(const boost::bad_lexical_cast &) {
					std::cout<<"\n bad --dictionaryLength value "<<argv[i+1];
					break;
				}
			}
			if(i==argc-1) {
				if(boost::filesystem::exists(argv[i])) {
					m_imageNames.push_back(argv[i]);
					std::cout<<"\n found image "<<argv[i];
					m_isValid = true;
				}
				else 
					std::cout<<"\n image doesn't exist "<<argv[i];
			}
		}
	}
	if(m_isValid) {
		std::cout<<"\n atom size "<<m_atomSize;
		std::cout<<"\n dictionary length "<<m_dictionaryLength;
		countPatches();
		m_isValid = m_numPatches > 0;
	}
}

LfParameter::~LfParameter() {}

bool LfParameter::isValid() const
{ return m_isValid; }

bool LfParameter::searchImagesIn(const char * dirname)
{
	m_imageNames.clear();
	if(!boost::filesystem::exists(dirname)) {
		std::cout<<"\n dir doesn't exist "<<dirname;
		return false;
	}
	std::cout<<"\n searching images in dir "<<dirname<<" ...";
	
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
				}
			}
		}
	}
	
	std::cout<<" found "<<m_imageNames.size();
	return true;
}

void LfParameter::countPatches()
{
    ZEXRImage img;
	m_numPatches = 0;
	std::vector<std::string >::const_iterator it = m_imageNames.begin();
	for(; it!=m_imageNames.end();++it) {
		std::string fn = *it;
        
		ZEXRImage::PrintChannelNames(fn);
		
		if(img.open(fn.c_str())) {
			m_numPatches += (img.getWidth() / m_atomSize) * (img.getHeight() / m_atomSize);
        }
		else 
			std::cout<<"\n cannot open exr "<<fn;
	}
	std::cout<<"\n num patch "<<m_numPatches;
}

int LfParameter::atomSize() const
{ return m_atomSize; }

int LfParameter::dictionaryLength() const
{ return m_dictionaryLength; }

std::string LfParameter::imageName(int i) const
{ return m_imageNames[i]; }

int LfParameter::numPatches() const
{ return m_numPatches; }

int LfParameter::dimensionOfX() const
{ return m_atomSize * m_atomSize * 3; }

int LfParameter::randomImageInd() const
{ 
	if(m_imageNames.size() < 2) return 0; 
	return rand() % m_imageNames.size();
}

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

ZEXRImage *LfParameter::openImage(const int ind)
{
    int idx;
    if(isImageOpened(ind, idx)) {
        return m_openedImages[idx]._image;
    }
    
    idx = m_currentImage;
	if(m_openedImages[idx]._ind < 0) {
		m_openedImages[idx]._image = new ZEXRImage;
		std::cout<<" create "<<m_openedImages[idx]._image<<"   ";
	}
	m_openedImages[idx]._ind = ind;
	m_openedImages[idx]._image->open(imageName(ind));
	//std::cout<<" open "<<m_openedImages[idx]._image<<"   ";
	m_currentImage = (m_currentImage + 1) % MAX_NUM_OPENED_IMAGES;
	return m_openedImages[idx]._image;
}

void LfParameter::PrintHelp()
{
	std::cout<<"\n lfr (Light Field Research) version 20151122"
	<<"\nUsage:\n lfr [option] [file]"
	<<"\nDescription:\n lfr learns the underlying pattern of input images."
	<<"\n Input file must be image of OpenEXR format. If no file is provided,"
	<<"\n current dir will be searched for any file with name ending in .exr."
	<<"\nOptions:\n -as or --atomSize    integer    size of image atoms, no less than 10"
	<<"\n -dl or --dictionaryLength    integer    length of dictionary, no less than 1024"
	<<"\n -h or --help    print this information"
	<<"\n";
}

}
