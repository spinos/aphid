#pragma once
class MlFeather;
class MlFeatherCollection;

class FeatherExample
{
public:
    FeatherExample();
	MlFeather * selectedExample() const;
    static MlFeatherCollection * FeatherLibrary;
private:
};
