#include <iostream>
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include <boost/regex.hpp>
using namespace std;
using namespace boost::filesystem;

char doMatch(const char* content, const char* expression)
{
	cout<<"match \""<<content<<"\" against \""<<expression<<"\"\n";
	const boost::regex re1(expression);
	
	std::string tomatch(content);
	std::string::const_iterator start, end;
    start = tomatch.begin();
    end = tomatch.end();
	boost::match_results<std::string::const_iterator> what;
	if(regex_match(tomatch, what, re1, boost::match_default) )
	{
		
		cout<<"  matched "<<what[0]<<endl;
		return 1;
	}
	cout<<"no match\n"<<endl;
	return 0;
}

void doSearch(const char* content, const char* expression)
{
	cout<<"search \""<<content<<"\" against \""<<expression<<"\"\n";
	
 const boost::regex re1(expression);
	
	std::string tomatch(content);
	std::string::const_iterator start, end;
    start = tomatch.begin();
    end = tomatch.end();
	boost::match_results<std::string::const_iterator> what;
	char found = 0;
	while(regex_search(start, end, what, re1, boost::match_default) )
	{
		
		cout<<"  found "<<what[0]<<endl;
		start = what[0].second;
		found = 1;
	}
	if(!found)
		cout<<"not found\n"<<endl;
}

void lsTypedFile()
{
	boost::match_results<std::string::const_iterator> what;
    
	cout<<"ls .cpp and .o files\n";
	static const boost::regex e("(.*)\\.o|(.*)\\.cpp");

  	  path head_path("/Users/jianzhang/aphid/booregex");
		directory_iterator end_iter;
		for ( directory_iterator itdir( head_path );
			  itdir != end_iter;
			  ++itdir )
		{
			if ( is_regular_file( itdir->status() ) )
			{
				
				regex_match(itdir->path().filename(), what, e, boost::match_partial );
				if(what[0].matched){
					
					cout<<"  "<<itdir->path().filename()<<endl;
				}
					
				
			}
		}
}

int
main() 
  {
	lsTypedFile();
	doSearch("|foo_a|foo_b|foo_c_grp", "\\|foo_");
	doMatch("|slow=|ig|e|f|g", "\\|.*\\=\\|.*");
	doMatch("-slow_=-ig_a_", "\\-.*\\=\\-.*");
	doSearch("avbc=0121=cds=907", "\\w+\\=\\d+");
	doSearch("{avbc=0121}={cds=907}", "\\{(\\w+|\\d+)\\}");
	doSearch("this is the log:\ndata 1-29-2011 change minor\ndata 12-11-2010 change significant", "\\d{1,2}-\\d{1,2}-\\d{4}");			
	doMatch("a_prefix_group2", "(.*)\\_group2");
	doSearch("{abc:0.032=2.3121}", "\\{(.*)\\:");
	doSearch("{ns:0=1.324e2}", "\\=.*\\}");
	doMatch("|2234|ere", "\\|.*");
	doMatch("{abc:0.5=0.3}", ".*\\:.*\\=.*");
	doMatch("{|a|b|c=|a|c|d}", ".*\\=.*");
	doSearch("{abc:0.032=2.3121}", "\\:(.*)\\=");
	doMatch("1c", "\\d[abc]");
	
    return 0;
  };
