#ifndef JUL_PARAMETER_H
#define JUL_PARAMETER_H

#include <string>
#include <vector>

namespace jul {

class Parameter {

	std::string m_outFileName;
public:
	enum OperationFlag {
		kUnknown = 0,
		kHelp = 1,
		kGenerate = 2
	};
	
	Parameter(int argc, char *argv[]);
	virtual ~Parameter();
	
	bool isValid() const;
	static void PrintHelp();
	
	OperationFlag operation() const;
	
	std::string outFileName() const;
protected:

private:
	OperationFlag m_opt;
};

}
#endif        //  #ifndef LFPARAMETER_H

