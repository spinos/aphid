#pragma once

class BaseState {
public:
    BaseState();
    
    void enable();
	void disable();
	bool isEnabled() const;
private:
    bool m_enabled;
};
