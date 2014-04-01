/*
 *  main.cpp
 *  
 *
 *  Created by jian zhang on 3/13/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */


#include <iostream>
#include <istream>
#include <ostream>
#include <string>
#include "boost/filesystem.hpp"
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/filesystem/convenience.hpp"
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/erase.hpp>
#include <boost/foreach.hpp>
#include <boost/regex.hpp>

static char seqName[] = "/Users/jianzhang/aphid/findseq/foo.1.ass";

using namespace boost::filesystem;
using namespace boost::algorithm;
using namespace std;

bool isFileInSequence(const path & full, const path & ref)
{
    const std::string pattern = (boost::format("%1%.-?\\d{1,4}%2%") % ref.stem().stem().string() % ref.extension().string()).str();
	//cout<<"pattern is "<<pattern<<"\n";
	//cout<<"compare to "<<full.filename()<<"\n";
	
	const boost::regex re1(pattern);
	boost::match_results<std::string::const_iterator> what;
	if(!regex_match(full.filename().string(), what, re1, boost::match_default))
		return false;
	
    return true;   
}

int extractFrameNumber(const path & full, const path & ref)
{
    string filename = full.filename().string();
    const string head = (boost::format("%1%.") % ref.stem().stem().string()).str();
    erase_first(filename, head);
    erase_last(filename, ref.extension().string());
    return atoi(filename.c_str());
}

int main(int argc, char* argv[])
{
	std::clog<<"find seq by "<<seqName<<"\n";
	path apath(seqName);
	std::clog<<"parent dir name is "<<apath.parent_path()<<"\n";
	cout<<"filename is "<<apath.filename()<<"\n";
	cout<<"stem is "<<apath.stem()<<"\n";

	
	typedef vector<path> vec;             // store paths,
    vec v;                                // so we can sort them later

    copy(directory_iterator(apath.parent_path()), directory_iterator(), back_inserter(v));

    sort(v.begin(), v.end()); 
    
    typedef vector<int> FrameList;
    FrameList frames;
    for (vec::const_iterator it (v.begin()); it != v.end(); ++it)
    {
        if(isFileInSequence(*it, apath.filename())) {
            cout << "   " << *it << '\n';
            frames.push_back(extractFrameNumber(*it, apath.filename()));
        }
    }
    
    sort(frames.begin(), frames.end());
    cout<<"num frames "<<frames.size()<<"\n";
    cout<<"range[ "<<frames.front()<<" , "<<frames.back()<<" ]\n";

	return 0;
}