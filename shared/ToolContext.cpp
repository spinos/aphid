#include <ToolContext.h>

ToolContext::ToolContext()
{
	setContext(SelectVertex);
	setPreviousContext(UnknownInteract);
}

void ToolContext::setContext(InteractMode val)
{
    m_ctx = val;
}

ToolContext::InteractMode ToolContext::getContext() const
{
    return m_ctx;
}

void ToolContext::setPreviousContext(InteractMode val)
{
	m_preCtx = val;
}
    
ToolContext::InteractMode ToolContext::previousContext() const
{
	return m_preCtx;
}