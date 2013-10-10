#include "BaseState.h"

BaseState::BaseState() 
{
    m_enabled = false;
}

void BaseState::enable()
{
	m_enabled = true;
}

void BaseState::disable()
{
	m_enabled = false;
}

bool BaseState::isEnabled() const
{
	return m_enabled;
}
