#pragma once
#include <SDL.h>
#include <vector>

struct MouseInformation
{
	int x, y, scrollwheel;
	bool lmb, rmb, mouse3;
};

class EventManager final
{
public:
	static void ProcessInputs(bool& isLooping, bool& takeScreenshot, float elapsedSec);
	static MouseInformation GetMouseInformation();

	static bool IsInput(SDL_EventType type, SDL_Keycode key);
	static bool IsKeyPressed(SDL_Keycode key);
	static bool IsKeyDown(SDL_Scancode key);

private:
	EventManager() = delete;
	~EventManager() = delete;

	static void KeyDownEvent(const SDL_KeyboardEvent& e);
	static void KeyUpEvent(const SDL_KeyboardEvent& e);
	static void MouseMotionEvent(const SDL_MouseButtonEvent& e);
	static void MouseDownEvent(const SDL_MouseButtonEvent& e);
	static void MouseUpEvent(const SDL_MouseButtonEvent& e);
	static void MouseWheelEvent(const SDL_MouseWheelEvent& e);

	static int m_ScrollWheelValue;
	static std::vector<SDL_Event> m_Events;
};