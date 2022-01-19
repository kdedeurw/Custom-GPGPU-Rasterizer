#pragma once
#include "SDL.h"
#include <vector>

union SDL_Event;
class EventManager
{
public:
	static void ProcessInputs(bool& isLooping, bool& takeScreenshot, float elapsedSec);
	static void GetMouseButtonStates(bool& isLmb, bool& isRmb);
	static void GetRelativeMouseValues(float& x, float& y);
	static void GetScrollWheelValue(int& scrollWheelValue);

private:
	EventManager() = default;
	virtual ~EventManager();

	static void KeyDownEvent(const SDL_KeyboardEvent& e);
	static void KeyUpEvent(const SDL_KeyboardEvent& e);
	static void MouseMotionEvent(const SDL_MouseButtonEvent& e);
	static void MouseDownEvent(const SDL_MouseButtonEvent& e);
	static void MouseUpEvent(const SDL_MouseButtonEvent& e);
	static void MouseWheelEvent(const SDL_MouseWheelEvent& e);

	static int m_ScrollWheelValue;
};