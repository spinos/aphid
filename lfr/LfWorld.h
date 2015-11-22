/*
 *  LfWorld.h
 *  
 *
 *  Created by jian zhang on 11/22/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <string>
#include <vector>

class LfParameter {

	std::vector<std::string > m_imageNames;
	std::string m_imageName;
	int m_atomSize;
	int m_dictionaryLength;
	bool m_isValid;
public:
	LfParameter(int argc, char *argv[]);
	virtual ~LfParameter();
	
	bool isValid() const;
	static void PrintHelp();
protected:

private:
	bool searchImagesIn(const char * dirname);
};

class LfWorld  {
	
	const LfParameter * m_param;
public:

	LfWorld(const LfParameter & param);
	virtual ~LfWorld();
	
	
protected:

private:
	
};