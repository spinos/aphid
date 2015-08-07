class AWorld
{
public:
    AWorld();
    virtual ~AWorld();
    
    virtual void stepPhysics(float dt);
    virtual void progressFrame();
protected:

private:

};
