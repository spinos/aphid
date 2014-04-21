/* getenv example: getting path */
#include <stdio.h>      /* printf */
#include <stdlib.h>     /* getenv */
#include <string>
#include <iostream>
#include "boost/filesystem.hpp"
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/filesystem/convenience.hpp"
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/erase.hpp>
#include <boost/foreach.hpp>
#include <boost/regex.hpp>
#include <boost/tokenizer.hpp>

bool isVarInSequence(const std::string & ref)
{
    const std::string pattern = ".*\\{\\$.+\\}.+";
    std::cout<<"match pattern is "<<pattern<<"\n";

	const boost::regex re1(pattern);
	boost::match_results<std::string::const_iterator> what;
	if(!regex_match(ref, what, re1, boost::match_default))
		return false;
	
    return true;   
}

bool searchVarInSequence(const std::string & ref, std::string & dst)
{
    const std::string pattern = ".*\\{\\$(.+)\\}.+";
    std::cout<<"search pattern is "<<pattern<<"\n";
    
    std::string::const_iterator start, end;
    start = ref.begin();
    end = ref.end();

	const boost::regex re1(pattern);
	boost::match_results<std::string::const_iterator> what;
	while(regex_search(start, end, what, re1, boost::match_default) ){
		for(unsigned i = 0; i <what.size(); i++)
		{
			std::cout<<str(boost::format(" %1% : %2% ") % i % what[i]);
			if(i==1) {
			    dst = what[i];
			    return true;
			}
		}
		start = what[0].second;
	}
	
    return false;   
}

bool firstInVarSeq(const std::string & varKey, std::string & dst)
{
    char* pPath;
    pPath = getenv (varKey.c_str());
    if(pPath==NULL) 
        return false;
    dst = str(boost::format("%1%") % pPath);
    
    typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep(";");
	tokenizer tokens(dst, sep);
	tokenizer::iterator tok_iter = tokens.begin();
	if(tok_iter != tokens.end())
	    dst = *tok_iter;
	
    return true;
}

std::string replaceVarInSequence(const std::string & ref, const std::string & varstr, const std::string & with)
{
	std::string out = ref;
	boost::algorithm::replace_all(out, str(boost::format("{$%1%}") % varstr), with);
	
    return out;   
}

int main ()
{
    const std::string insq("{$MAYA_MODULE_PATH}/abc");
    if(isVarInSequence(insq)) std::cout<<insq<<" is var\n";
    else std::cout<<insq<<" is NOT var\n";
    
    std::string varstr;
    if(searchVarInSequence(insq, varstr))
        std::cout<<"var is "<<varstr<<"\n";
    
    std::string varrepstr;
    if(firstInVarSeq(varstr, varrepstr))
        std::cout<<"replace {$"<<varstr<<"} with "<<varrepstr<<"\n";
    
    std::cout<<replaceVarInSequence(insq, varstr, varrepstr);
    return 0;
}
