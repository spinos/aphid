#ifndef EXRS_PARAMETER_H
#define EXRS_PARAMETER_H

#include <string>
#include <vector>

namespace exrs {

class Parameter {
	
	std::string m_inFileName;
	
public:
	enum Operation {
		kHelp = 0,
		kTestSampler = 1
	};
	
	Parameter(int argc, char *argv[]);
	virtual ~Parameter();
	
	Operation operation() const;
	const std::string & inFileName() const;
	static void PrintHelp();
	
protected:

private:
	Operation m_operation;
};

}
#endif        //  #ifndef LFPARAMETER_H

