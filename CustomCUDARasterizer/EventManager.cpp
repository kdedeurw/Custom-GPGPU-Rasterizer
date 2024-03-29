#include "PCH.h"
#include "EventManager.h"

int EventManager::m_ScrollWheelValue{};
std::vector<SDL_Event> EventManager::m_Events{};

void EventManager::ProcessInputs(bool& isLooping, bool& takeScreenshot, float elapsedSec)
{
	m_Events.clear();
	m_ScrollWheelValue = 0;
	SDL_Event e;
	while (SDL_PollEvent(&e) != 0)
	{
		m_Events.push_back(e);
		switch (e.type)
		{
		case SDL_QUIT:
			isLooping = false;
			break;
		case SDL_KEYDOWN:
			if (e.key.keysym.sym == SDLK_ESCAPE)
			{
				bool isRelativeMouse = SDL_GetRelativeMouseMode();
				isRelativeMouse = !isRelativeMouse;
				SDL_SetRelativeMouseMode((SDL_bool)isRelativeMouse);
			}
			KeyDownEvent(e.key);
			break;
		case SDL_KEYUP:
			if (e.key.keysym.scancode == SDL_SCANCODE_X)
			{
				takeScreenshot = true;
			}
			KeyUpEvent(e.key);
			break;
		case SDL_MOUSEMOTION:
			MouseMotionEvent(e.button);
			break;
		case SDL_MOUSEBUTTONDOWN:
			MouseDownEvent(e.button);
			break;
		case SDL_MOUSEBUTTONUP:
			MouseUpEvent(e.button);
			break;
		case SDL_MOUSEWHEEL:
			MouseWheelEvent(e.wheel);
			break;
		}
	}
}

void EventManager::KeyDownEvent(const SDL_KeyboardEvent & e)
{
}

void EventManager::KeyUpEvent(const SDL_KeyboardEvent & e)
{
}

void EventManager::MouseMotionEvent(const SDL_MouseButtonEvent & e)
{
}

void EventManager::MouseDownEvent(const SDL_MouseButtonEvent & e)
{
	switch (e.button)
	{
	case SDL_BUTTON_LEFT:
		break;
	case SDL_BUTTON_RIGHT:
		break;
	case SDL_BUTTON_MIDDLE:
		break;
	}
}

void EventManager::MouseUpEvent(const SDL_MouseButtonEvent & e)
{
	switch (e.button)
	{
	case SDL_BUTTON_LEFT:
		break;
	case SDL_BUTTON_RIGHT:
		break;
	case SDL_BUTTON_MIDDLE:
		break;
	}
}

void EventManager::MouseWheelEvent(const SDL_MouseWheelEvent & e)
{
	m_ScrollWheelValue = e.y;
}

MouseInformation EventManager::GetMouseInformation()
{
	MouseInformation mi{};
	const Uint32 buttons = SDL_GetRelativeMouseState(&mi.x, &mi.y);
	mi.lmb = buttons & SDL_BUTTON_LMASK;
	mi.rmb = buttons & SDL_BUTTON_RMASK;
	mi.mouse3 = buttons & SDL_BUTTON_MMASK;
	mi.scrollwheel = m_ScrollWheelValue;
	return mi;
}

bool EventManager::IsInput(SDL_EventType type, SDL_Keycode key)
{
	//TODO: command listeners lmao
	for (const SDL_Event& e : m_Events)
	{
		if (e.type == type)
		{
			return e.key.keysym.sym == key;
		}
	}
	return false;
}

bool EventManager::IsKeyPressed(SDL_Keycode key)
{
	for (const SDL_Event& e : m_Events)
	{
		if (e.type == SDL_KEYDOWN)
		{
			return e.key.keysym.sym == key;
		}
	}
	return false;
}

bool EventManager::IsKeyDown(SDL_Scancode key)
{
	const uint8_t* pKeyboardState = SDL_GetKeyboardState(0);
	return pKeyboardState[key];
}