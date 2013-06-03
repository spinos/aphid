#pragma once

class ToolContext {
public:
    enum InteractMode {
        UnknownInteract = 0,
        SelectVertex = 1,
        SelectEdge = 2,
        SelectFace = 3
    };
    
    enum ActionRank {
        UnknownAction = 0,
        SetWaleEdge = 1
    };
    
    ToolContext();
    
    void setContext(InteractMode val);
    InteractMode getContext() const;
    
private:
    InteractMode m_ctx;
};
