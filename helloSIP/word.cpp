#include "word.h"

#include <iostream>
#include <sstream>

using namespace std;
namespace MY {
Word::Word(const char *w)
{
	_the_word = w;
}

const char *Word::reverse() const
{
	string r(_the_word);
	int count = r.size();

	for(int i = 0; i < count; i++) {
		r[i] = _the_word[count - 1  - i];	
	}
	
	return r.c_str();
}
};
