// Define the interface to the word library.

class FooImage {
	const char* _the_word;
	float _red;
	char* _data;
	float* _color;
	int _img_w, _img_h;

public:
    FooImage(const char *w);
    
    void setSize(int w, int h);
    void addColor(void *d);

    void *display();
    
    void setRed(float red);
};
