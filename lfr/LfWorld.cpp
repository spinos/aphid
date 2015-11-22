/*
 *  LfWorld.cpp
 *  
 *
 *  Created by jian zhang on 11/22/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>
#include "LfWorld.h"
#include <boost/lexical_cast.hpp>
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/filesystem/convenience.hpp"
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/case_conv.hpp>
#include <zEXRImage.h>

LfParameter::LfParameter(int argc, char *argv[])
{
	std::cout<<"\n lfr (Light Field Research) version 20151122";
	m_isValid = false;
	m_atomSize = 10;
	m_dictionaryLength = 256;
	bool foundImages = false;
	if(argc == 1) {
		m_isValid = searchImagesIn("./");
	}
	else {
		int i = 1;
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
					if(m_dictionaryLength < 256) {
						std::cout<<"\n bad --dictionaryLength value (< 256)";
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
					m_imageNames.push_back( boost::filesystem::basename(itdir->path()));
				}
			}
		}
	}
	
	std::cout<<" found "<<m_imageNames.size();
	return true;
}

void LfParameter::countPatches()
{
	m_numPatches = 0;
	std::vector<std::string >::const_iterator it = m_imageNames.begin();
	for(; it!=m_imageNames.end();++it) {
		std::string fn = *it + ".exr";
		
		ZEXRImage img;
		if(img.open(fn.c_str()))
			m_numPatches += (img.getWidth() / m_atomSize) * (img.getHeight() / m_atomSize);
		else 
			std::cout<<"\n cannot open exr "<<fn;
	}
	std::cout<<"\n num patch "<<m_numPatches;
}

void LfParameter::PrintHelp()
{
	std::cout<<"\n lfr (Light Field Research) version 20151122"
	<<"\nUsage:\n lfr [option] [file]"
	<<"\nDescription:\n lfr learns the underlying pattern of input images."
	<<"\n Input file must be image of OpenEXR format. If no file is provided,"
	<<"\n current dir will be searched for any file with name ending in .exr."
	<<"\nOptions:\n -as or --atomSize    integer    size of image atoms, no less than 10"
	<<"\n -dl or --dictionaryLength    integer    length of dictionary, no less than 256"
	<<"\n -h or --help    print this information"
	<<"\n";
}

LfWorld::LfWorld(const LfParameter & param) 
{
	m_param = &param;
}

LfWorld::~LfWorld() {}
