#include <ToolContext.h>

ToolContext::ToolContext()
{
	setContext(SelectVertex);
}

void ToolContext::setContext(InteractMode val)
{
    m_ctx = val;
}

ToolContext::InteractMode ToolContext::getContext() const
{
    return m_ctx;
}
