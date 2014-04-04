// Define the interface to the word library.
#include <string>
#include <vector>
namespace MY {
    
class Ooops : public std::exception {
public:
  const char* what() const throw() {return "Ooops!\n";}
};

class Word {
	const char* the_word;

public:
    Word();

    std::string reversed(const std::string & w) const;
	std::string str(const std::string & name) const;
	std::string strvec(const std::vector<std::string> & name) const;
	int someerr() const;
};
};
