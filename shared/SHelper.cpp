#include "SHelper.h"
#include <stdlib.h>
#include <time.h>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <string>
#include <vector>
#include <sstream>
#include <boost/regex.hpp>
#include <boost/format.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/algorithm/string.hpp>
using namespace std;

void SHelper::divideByFirstSpace(std::string& ab2a, std::string& b)
{
	b = ab2a;
	int found = ab2a.find(' ', 0);
	if(found < 0) return;
		
	ab2a.erase(found);
	b.erase(0, found);
}

void SHelper::trimByFirstSpace(std::string& res)
{
	int found = res.find(' ', 0);
	if(found < 0) return;
		
	res.erase(found);
}

void SHelper::getTime(std::string& log)
{
	time_t rawtime;
	struct tm * timeinfo;
	
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	log = std::string(asctime(timeinfo));
}

void SHelper::cutByFirstDot(std::string& res)
{
	int found = res.find('.', 0);
	if(found < 0) return;
		
	res.erase(found);
}

void SHelper::cutByLastDot(std::string& res)
{
	int found = res.rfind('.', res.size());
	if(found < 0) return;
		
	res.erase(found);
}

void SHelper::cutByLastSlash(std::string& res)
{
	int found = res.rfind('\\', res.size());
	if(found > 1) 
	{
		found++;
		res.erase(found);
		return;
	}	
	
	found = res.rfind('/', res.size());
	if(found > 1) 
	{
		found++;
		res.erase(found);
	}
	return;
}

void SHelper::changeFrameNumber(std::string& res, int frame)
{
	int first = res.find('.', 0);
	if(first < 0) return;
		
	int last = res.rfind('.', res.size());
	if(last < 0) return;
	
	char mid[8];
	sprintf(mid, ".%d.", frame);
	
	res.erase(first, last-first+1);
	res.insert(first, mid);
}

void SHelper::changeFrameNumber(std::string& res, int frame, int padding)
{
	int first = res.find('.', 0);
	if(first < 0) return;
		
	int last = res.rfind('.', res.size());
	if(last < 0) return;
	
	std::stringstream sst;
	sst<<frame;
	const std::string number = sst.str();
	int zerosToFill = padding - number.size();
	if(zerosToFill < 0) zerosToFill = 0;
	
	sst.str("");
	sst<<'.';
	for(int i=0; i<zerosToFill; i++) sst<<'0';
	sst<<number;
	sst<<'.';
	
	res.erase(first, last-first+1);
	res.insert(first, sst.str());
}

int SHelper::safeConvertToInt(const double a)
{
	double b = (a*100.0 + 0.5)/100.0;
	return (int)b;
}

int SHelper::getFrameNumber(std::string& name)
{
	int first_dot = name.find_first_of('.', 0);
	first_dot++;
	int last_dot = name.find_last_of('.', name.size()-1);
	last_dot--;
	
	if(last_dot < first_dot) return 0;
	
	std::string sub = name.substr(first_dot, last_dot - first_dot + 1);
	
	return atoi(sub.c_str());
}

int SHelper::compareFilenameExtension(std::string& name, const char* other)
{
	int found = name.find_last_of('.', name.size()-1);
	if(found < 0) return 0;
	found++;
	if(name.compare(found, 6, std::string(other)) == 0) return 1;
	
	return 0;
}

void SHelper::setFrameNumberAndExtension(std::string& name, int frame, const char* extension)
{
	int found = name.find_first_of('.', 0);
	if(found > 0) name.erase(found);
		
	char buf[8];
	sprintf(buf, ".%d.", frame);
	name.append(buf);
	name.append(extension);
}

int SHelper::getFrameNumber(const std::string& name)
{
	std::string sbuf = name;
	int found = sbuf.find_last_of('.', sbuf.size()-1);
	if(found < 0) return 0;
	sbuf.erase(found);
	
	found = sbuf.find_last_of('.', sbuf.size()-1);
	if(found < 0) return 0;
	sbuf.erase(0, found);

	int res;
	sscanf(sbuf.c_str(), "%i", &res);
	return res;
}

void SHelper::removeFilenameExtension(std::string& name)
{
	int found = name.find_last_of('.', name.size()-1);
	if(found < 0) return;
	
	name.erase(found);
}


void SHelper::changeFilenameExtension(std::string& name, const char* ext)
{
	int found = name.find_last_of('.', name.size()-1);
	if(found < 0) return;
	found++;
	
	name.erase(found);
	name.append(ext);
}

void SHelper::validateFilePath(std::string& name)
{
	std::string str(name);

	int found = str.find('|', 0);
	
	while(found>-1)
	{
		str[found] = '_';
		found = str.find('|', found);
	}
	
	found = str.find(':', 0);
	
	while(found>-1)
	{
		str[found] = '_';
		found = str.find(':', found);
	}
		
	name = std::
	
	
	string(str.c_str());
}

void SHelper::replacefilename(std::string& res, std::string& toreplace)
{
	int founddot;
	int foundlash = res.rfind('\\', res.size());
	if(foundlash > 1) 
	{
		foundlash++;
		founddot = res.find('.', foundlash);
		if(founddot > foundlash)
		{
			res.erase(foundlash, founddot-foundlash);
			res.insert(foundlash, toreplace);
		}
		return;
	}	
	
	foundlash = res.rfind('/', res.size());
	if(foundlash > 1) 
	{
		foundlash++;
		founddot = res.find('.', foundlash);
		if(founddot > foundlash)
		{
			res.erase(foundlash, founddot-foundlash);
			res.insert(foundlash, toreplace);
		}
	}
}

void SHelper::findLastAndReplace(std::string& res, const char *tofind, const char *toreplace)
{
	std::string r = res;
	boost::algorithm::replace_last(r, tofind, toreplace);
	res = r;
}

void SHelper::cutfilepath(std::string& res)
{
	int foundf = res.rfind('\\', res.size());
	if(foundf > 1) 
	{
		foundf++;
	}	
	
	int foundb = res.rfind('/', res.size());
	if(foundb > 1) 
	{
		foundb++;
	}
	
	if(foundb > foundf)
		foundf = foundb;
		
	if(foundf > 1)
		res.erase(0, foundf);
}

void SHelper::changeFrameNumberFistDot4Digit(std::string& res, int frame)
{
	int first = res.find('.', 0);
	if(first < 0) return;
	
	char mid[8];
	if(frame<10) sprintf(mid, ".000%d.", frame);
	else if(frame<100) sprintf(mid, ".00%d.", frame);
	else if(frame<1000) sprintf(mid, ".0%d.", frame);
	else sprintf(mid, ".%d.", frame);
	
	res.erase(first, 6);
	res.insert(first, mid);
}

//#include <maya/MGlobal.h>

char SHelper::isInArrayDividedBySpace(const char* handle, const char* array)
{
	std::string full = array;
	int start = 0;
	int end = full.find(' ', start);
	std::string frag;
	// no space
	if(end < 0) {
		frag = array;
		if(frag.compare(handle)==0) return 1;
		else return 0;
	}

	
	while(end > start) {
		frag = full.substr(start, end - start);
		
		//MGlobal::displayInfo(MString(frag.c_str())+MString("start ")+start+ " end" + end);
		
		if(frag.compare(handle) == 0) return 1;
		start = end+1;
		end = full.find(' ', start);
		
		if(end < 0) {// last one
			frag = full.substr(start);
			if(frag.compare(handle) == 0) return 1;
		}
	}
	return 0;
}

void SHelper::filenameWithoutPath(std::string& res)
{
	cutByLastDot(res);
	int found = res.rfind('\\', res.size());
	if(found > 1) 
	{
		found++;
		res = res.substr(found);
		return;
	}	
	
	found = res.rfind('/', res.size());
	if(found > 1) 
	{
		found++;
		res = res.substr(found);
	}
}

void SHelper::protectComma(std::string& res)
{
	res.insert(res.size()-1, "\\");
	res.insert(0, "\\");
}

void SHelper::ribthree(std::string& res)
{
	res.erase(0, 6);
	res.erase(res.size()-1, 1);
	int found = res.find(',', 0);
	while(found>-1) {
		res[found] = ' ';
		found = res.find(',', found);
	}
}

int SHelper::findPartBeforeChar(std::string& full, std::string& frag, int start, char sep)
{
	int found = full.find(sep, start);
	if(found < 0  ) frag = full;
#ifndef WIN32
	else if(found == full.size() - 1) frag = full.substr(start, full.size() - start);
#endif
	else frag = full.substr(start, found - start -1);
	return found;
}

void SHelper::protectCommaFree(std::string& res)
{
	int lstart = 0;
	int lend = res.find('\"', lstart);
	while(lend > 0) {
		res.insert(lend, "\\");
		lstart = lend+2;
		lend = res.find('\"', lstart);
	}
}

void SHelper::endNoReturn(std::string& res)
{
	int end = res.size() - 1;
	if( res[end] == '\n' || res[end] == '\r') res.erase(end);
}

void SHelper::listAllNames(std::string& name, std::vector<std::string>& all)
{
	all.clear();
	std::string r;
	std::string str = name;
	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep("|/");
	tokenizer tokens(str, sep);
	for (tokenizer::iterator tok_iter = tokens.begin();
		tok_iter != tokens.end(); ++tok_iter)
		{
// mesh shape will be the last
			tokenizer::iterator nextit = tok_iter;
			++nextit;
			r = r + "|" +(*tok_iter);
			all.push_back(r);	
		}
}

char SHelper::hasParent(std::string& name)
{
	std::vector<std::string> parents;
	SHelper::listParentNames(name, parents);
	if(parents.size() < 1)
		return 0;
	return 1;
}

void SHelper::listParentNames(const std::string& name, std::vector<std::string>& parents)
{
	parents.clear();
	std::string r;
	std::string str = name;
	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep("|");
	tokenizer tokens(str, sep);
	for (tokenizer::iterator tok_iter = tokens.begin();
		tok_iter != tokens.end(); ++tok_iter)
		{
// mesh shape will be the last
			tokenizer::iterator nextit = tok_iter;
			++nextit;
			if(nextit != tokens.end())
			{
				r = r + "|" +(*tok_iter);
				parents.push_back(r);
			}
		}
}

std::string SHelper::getHighestParentName(std::string& name)
{
	std::string r;
	std::string str = name;
	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep("|");
	tokenizer tokens(str, sep);
	tokenizer::iterator tok_iter = tokens.begin();
	r = "|" +(*tok_iter);
	return r;
}

void SHelper::getHierarchy(const char *name, std::vector<std::string> &res)
{
	res.clear();
	std::string r;
	std::string str = name;
	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep("|");
	tokenizer tokens(str, sep);
	for (tokenizer::iterator tok_iter = tokens.begin();
		tok_iter != tokens.end(); ++tok_iter)
		{
			r = r + "|" +(*tok_iter);
			res.push_back(r);
		}
}

std::string SHelper::getParentName(const std::string& name)
{
	std::string r("");
	std::string str = name;
	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep("|/");
	tokenizer tokens(str, sep);
	for (tokenizer::iterator tok_iter = tokens.begin();
		tok_iter != tokens.end(); ++tok_iter)
		{
// mesh shape will be the last
			tokenizer::iterator nextit = tok_iter;
			++nextit;
			if(nextit != tokens.end())
				r = r + "|" +(*tok_iter);
		}
	return r;
}

std::string SHelper::getLastName(const std::string& name)
{
	std::string r;
	std::string str = name;
	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep("|/");
	tokenizer tokens(str, sep);
	for (tokenizer::iterator tok_iter = tokens.begin();
		tok_iter != tokens.end(); ++tok_iter)
		{
// mesh shape will be the last
			r = (*tok_iter);
		}
	return r;
}

void SHelper::pathToFilename(std::string& name)
{
	std::string r;
	std::string str = name;
	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep("/:\\");
	tokenizer tokens(str, sep);
	for (tokenizer::iterator tok_iter = tokens.begin();
		tok_iter != tokens.end(); ++tok_iter)
		{
// mesh shape will be the last
			if(tok_iter != tokens.begin())
				r = r + "_";
			r = r +(*tok_iter);
		}
	name = r;
}

void SHelper::noColon(std::string& name)
{
	std::string r;
	std::string str = name;
	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep(":");
	tokenizer tokens(str, sep);
	for (tokenizer::iterator tok_iter = tokens.begin();
		tok_iter != tokens.end(); ++tok_iter)
		{
// mesh shape will be the last
			if(tok_iter != tokens.begin())
				r = r + "_";
			r = r +(*tok_iter);
		}
	name = r;
}

void SHelper::pathDosToUnix(std::string& name)
{
        size_t startNetwork = name.find("\\\\");
        if(startNetwork != 0)
                startNetwork = name.find("//");
        
	std::string r;
	if(startNetwork == 0) r = "//";
	
	std::string str = name;
	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep("\\");
	tokenizer tokens(str, sep);
	for (tokenizer::iterator tok_iter = tokens.begin();
		tok_iter != tokens.end(); ++tok_iter)
		{
// append / except last one
			tokenizer::iterator nextit = tok_iter;
			++nextit;
	
			r = r +(*tok_iter);
			
			if(nextit != tokens.end())
				r = r + "/";
		}
	name = r;
}

void SHelper::pathUnixToDos(std::string& name)
{
	size_t startNetwork = name.find("\\\\");
        if(startNetwork != 0)
                startNetwork = name.find("//");
        
	std::string r;
	if(startNetwork == 0) r = "\\\\";
	
	std::string str = name;
	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep("/");
	tokenizer tokens(str, sep);
	for (tokenizer::iterator tok_iter = tokens.begin();
		tok_iter != tokens.end(); ++tok_iter)
		{
// append \ except last one
			tokenizer::iterator nextit = tok_iter;
			++nextit;
	
			r = r +(*tok_iter);
			
			if(nextit != tokens.end())
				r = r + "\\";
		}
	name = r;
}

void SHelper::removePrefix(std::string& name)
{
	std::string r;
	std::string str = name;
	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep("/:\\");
	tokenizer tokens(str, sep);
	for (tokenizer::iterator tok_iter = tokens.begin();
		tok_iter != tokens.end(); ++tok_iter)
		{
			r = (*tok_iter);
		}
	name = r;
}

void SHelper::removeNodeName(std::string& name)
{
	std::string r;
	std::string str = name;
	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep(".");
	tokenizer tokens(str, sep);
	for (tokenizer::iterator tok_iter = tokens.begin();
		tok_iter != tokens.end(); ++tok_iter)
		{
			if(tok_iter != tokens.begin())
			{
				r += (*tok_iter);
				tokenizer::iterator nextit = tok_iter;
				++nextit;
				if(nextit != tokens.end())
					r += ".";
			}
		}
	name = r;
}

void SHelper::behead(std::string& name, const std::string& head)
{
	if(head.size() < 1)
		return;
	if(name.find(head) != 0)
		return;
		
	name = name.substr(head.size());
}

void SHelper::stripAll(std::string& name)
{
	std::string r;
	std::string str = name;
	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep(":_");
	tokenizer tokens(str, sep);
	for (tokenizer::iterator tok_iter = tokens.begin();
		tok_iter != tokens.end(); ++tok_iter)
		{
			name = (*tok_iter);
		}
}

std::string SHelper::afterLastUnderscore(const std::string &res)
{
	int found = res.rfind('_')+1;

	if(found < 0) return res;
	
	std::string result = res;
	
	result = result.erase(0, found);

	return result;
}

char SHelper::fuzzyMatch(std::string &one,std::string &another)
{
	std::string str_one=one;
	std::string str_another = another;
	int numTokenOne,numTokenAnother;
	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
    boost::char_separator<char> sep_one("|");
    tokenizer tokenOne(str_one, sep_one);
	numTokenOne=0;
	std::vector<std::string> one_tokens;
    for (tokenizer::iterator iterTokenOne = tokenOne.begin();iterTokenOne != tokenOne.end(); ++iterTokenOne)
	{
		one_tokens.push_back(*iterTokenOne);
       numTokenOne++; 
	}

	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
    boost::char_separator<char> sep_another("|");
    tokenizer tokenAnother(str_another, sep_another);
	numTokenAnother=0;
	std::vector<std::string> another_tokens;
    for (tokenizer::iterator iterTokenAnother = tokenAnother.begin();iterTokenAnother != tokenAnother.end(); ++iterTokenAnother)
	{
		another_tokens.push_back(*iterTokenAnother);
       numTokenAnother++;
	}
	// one should not have less number of blocks than another has  
	if(numTokenOne < numTokenAnother)
	{
		return 0;
	}
	else
	{
		 int numSpare=numTokenOne-numTokenAnother;
	     int blockIdOne=0;

		   for (tokenizer::iterator iterTokenOne = tokenOne.begin();iterTokenOne != tokenOne.end(); ++iterTokenOne)
			{
				int blockIdMinusSpare = blockIdOne - numSpare;
				// start matching at block[numSpare], and ignore all blocks before it
				if(blockIdMinusSpare > -1)
				{
					std::string curTokenOne = one_tokens[blockIdOne]; //getBlockFromTokensById(blockIdOne, tokenOne);
					curTokenOne=afterLastUnderscore(curTokenOne);
					std::string curTokenAnother = another_tokens[blockIdMinusSpare]; //getBlockFromTokensById(blockIdMinusSpare, tokenAnother);
					curTokenAnother=afterLastUnderscore(curTokenAnother);
					if(curTokenOne != curTokenAnother)
			    		return 0;
				}
			      blockIdOne++;	 
	         }

	}

	return 1;
    
}
//remove namespace with ':'
std::string SHelper::removeNamespace(const std::string &in)
{
	int found = in.rfind(':');

	if(found < 0) return in;
	
	std::string result = in;
	
	result = result.erase(0, found + 1);

	return result;
}
//get nsp from res
std::string getNamespace(const std::string &res)
{
	int found = res.rfind(':');

	if(found < 0) return res;
	
	std::string result = res;

	int length=res.length();
	
	result = result.erase(found,length);

	return result;

}

char SHelper::removeAnyNamespace(std::string &name)
{
	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep("|");
	tokenizer tokens(name, sep);

	std::stringstream sst;
	sst.str("");

	for (tokenizer::iterator tok_iter = tokens.begin();tok_iter != tokens.end(); ++tok_iter)
	{
		std::string curTokenname=(*tok_iter);
		
		std::string name_wo_namespace = removeNamespace(curTokenname);
		sst << "|" << name_wo_namespace; 
		
	}
	
	name = sst.str();
   	return 1;
}

// for all blocks, replace whatever before ':' with input toReplace
// if no ':' is found, add toReplace before ':'
char SHelper::replaceAnyNamespace(std::string &name,std::string &toReplace)
{
	if(toReplace == "")
		return 1;

	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep("|");
	tokenizer tokens(name, sep);

	name = "";

	for (tokenizer::iterator tok_iter = tokens.begin();tok_iter != tokens.end(); ++tok_iter)
	{
		std::string name_wo_namespace;
		std::string curTokenname=(*tok_iter);
		
	    	//animation with nps
	      name_wo_namespace=removeNamespace(curTokenname);
		  name +="|"+ toReplace + ":" + name_wo_namespace; 
		
	}
   	return 1;
}

char SHelper::fuzzyMatchNamespace(std::string &one, std::string &another)
{
	std::string str_one=one;
	std::string str_another=another;

	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
    boost::char_separator<char> sep_one("|");
    tokenizer tokenOne(str_one, sep_one);

	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
    boost::char_separator<char> sep_another("|");
    tokenizer tokenAnother(str_another, sep_another);

	for(tokenizer::iterator iterTokenOne = tokenOne.begin();iterTokenOne != tokenOne.end(); ++iterTokenOne)
	{
		for(tokenizer::iterator iterTokenAnother = tokenAnother.begin();iterTokenAnother != tokenAnother.end(); ++iterTokenAnother)
		{
			std::string curTokenOne=(*iterTokenOne);
			std::string curTokenAnother=(*iterTokenAnother);

			curTokenOne=removeNamespace(curTokenOne);
			curTokenAnother=removeNamespace(curTokenAnother);
			
			if(curTokenOne!=curTokenAnother)
				  return 0;
		}
	}

	return 1;
}

int SHelper::countColons(const std::string &in)
{
	int found = in.find(':');
	int count = 0;
	while(found > 0)
	{
		found = in.find(':', found + 1);
		count++;
	}
	return count;
}

void SHelper::validateUnixPath(std::string& name)
{
    boost::algorithm::replace_all(name, "\\", "/");
    boost::trim(name);
}

char SHelper::isMatched(const std::string &one, const std::string &another)
{
    if(one == another)
        return 1;

    std::string curnamewons = one;
    std::string terminalwons = another;
    
    curnamewons = SHelper::removeNamespace(curnamewons);
    terminalwons = SHelper::removeNamespace(terminalwons);
    
    if(curnamewons == terminalwons)
        return 1;
    
    std::string curnamewocl = one;
    std::string terminalwocl = another;
    
    SHelper::noColon(curnamewocl);
    SHelper::noColon(terminalwocl);
    
    if(curnamewocl == terminalwocl)
        return 1;
    
    return 0;
}

bool SHelper::IsPullPath(const std::string & name)
{ return (name[0] == '|' || name[0] == '/'); }

std::string SHelper::ParentPath(const std::string & name, const std::string & separator)
{
    std::string r("");
	std::string str = name;
	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep("|/");
	tokenizer tokens(str, sep);
	for (tokenizer::iterator tok_iter = tokens.begin();
		tok_iter != tokens.end(); ++tok_iter)
		{
// mesh shape will be the last
			tokenizer::iterator nextit = tok_iter;
			++nextit;
			if(nextit != tokens.end())
				r = r + separator +(*tok_iter);
		}
	return r;
}
//:~
