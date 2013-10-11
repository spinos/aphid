#pragma once

class BaseState {
public:
    BaseState();
    
    virtual void enable();
	virtual void disable();
	bool isEnabled() const;
private:
    bool m_enabled;
};
