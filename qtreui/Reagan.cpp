/*
 *  Reagan.cpp
 *  reui
 *
 *  Created by jian zhang on 12/15/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "Reagan.h"
#include <boost/regex.hpp>
#include <boost/format.hpp>

Reagan::Reagan() {}
Reagan::~Reagan() {}

void Reagan::runReReplace(std::string &result, const std::string &expression, const std::string &format)
{
	const boost::regex pattern(expression.c_str());
	result = boost::regex_replace(result,
						pattern,
						format, boost::match_default | boost::format_all);
}

void Reagan::validateUnixPath(std::string &result)
{
	Reagan::runReReplace(result, "\\\\", "/");
	Reagan::runReReplace(result, "(?<=[^^])/+", "(?1)/");
}

void Reagan::removeNamespaceInFullPathName(std::string &result)
{
	Reagan::runReReplace(result, "\\|\\w+:", "\\|");
}