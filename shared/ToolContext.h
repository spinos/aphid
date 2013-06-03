#pragma once

class ToolContext {
public:
    enum InteractMode {
        Unknown = 0,
        SelectVertex = 1,
        SelectEdge = 2,
        SelectFace = 3
    };
    
    ToolContext();
    
    void setContext(InteractMode val);
    InteractMode getContext() const;
private:
    InteractMode m_ctx;

};
