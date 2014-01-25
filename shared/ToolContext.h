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
		EraseBodyContourFeather = 6,
		ScaleBodyContourFeather = 7,
		PitchBodyContourFeather = 8,
		MoveInUV = 9,
		MoveVertexInUV = 10,
		MoveTransform = 11,
		RotateTransform = 12,
		MoveMeshComponent = 13,
		SmoothSurface = 14,
		SelectByColor = 15,
		Deintersect = 16,
		PaintMap = 17
    };
    
    enum ActionRank {
        UnknownAction = 0,
        SetWaleEdge = 1,
        SetSingleWaleEdge = 2,
		IncreaseWale = 3,
		DecreaseWale = 4,
		IncreaseCourse = 5,
		DecreaseCourse = 6,
		RebuildBodyContourFeather = 7,
		ClearBodyContourFeather = 8,
		AddFeatherExample = 9,
		RemoveFeatherExample = 10,
		IncreaseFeathExampleNSegment = 11,
		DecreaseFeathExampleNSegment = 12,
		LoadImage = 13,
		BakeAnimation = 14,
		AddSkeletonPose = 15,
		SaveSkeletonPose = 16,
		LaunchRender = 17,
		ChangeFeatherType = 18
    };
    
    ToolContext();
    
    void setContext(InteractMode val);
    InteractMode getContext() const;
	
	void setPreviousContext(InteractMode val);
    InteractMode previousContext() const;
    
private:
    InteractMode m_ctx, m_preCtx;
};
