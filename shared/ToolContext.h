#pragma once

class ToolContext {
public:
    enum InteractMode {
        UnknownInteract = 0,
        SelectVertex = 1,
        SelectEdge = 2,
        SelectFace = 3,
		CreateBodyContourFeather = 4,
		CombBodyContourFeather = 5,
		EraseBodyContourFeather = 6
    };
    
    enum ActionRank {
        UnknownAction = 0,
        SetWaleEdge = 1,
        SetSingleWaleEdge = 2,
		IncreaseWale = 3,
		DecreaseWale = 4,
		IncreaseCourse = 5,
		DecreaseCourse = 6
    };
    
    ToolContext();
    
    void setContext(InteractMode val);
    InteractMode getContext() const;
	
	void setPreviousContext(InteractMode val);
    InteractMode previousContext() const;
    
private:
    InteractMode m_ctx, m_preCtx;
};
