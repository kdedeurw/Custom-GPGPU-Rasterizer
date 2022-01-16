#pragma once
#include "SDL.h"

class EventManager
{
public:
	static EventManager* GetInstance()
	{
		if (!m_pEventManager) m_pEventManager = new EventManager{};
		return m_pEventManager;
	}
	~EventManager();

	void ProcessInputs(bool& isLooping, bool& takeScreenshot, float elapsedSec);
	void KeyDownEvent(const SDL_KeyboardEvent& e);
	void KeyUpEvent(const SDL_KeyboardEvent& e);
	void MouseMotionEvent(const SDL_MouseButtonEvent& e);
	void MouseDownEvent(const SDL_MouseButtonEvent& e);
	void MouseUpEvent(const SDL_MouseButtonEvent& e);
	void MouseWheelEvent(const SDL_MouseWheelEvent& e);

	void GetRelativeMouseValues(float& x, float& y);

private:
	static EventManager* m_pEventManager;
	EventManager() = default;

	bool m_IsRelativeMouse = true;
	bool m_IsLMB{}, m_IsRMB{}, m_IsMouse3{};
};

