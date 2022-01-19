#include "PCH.h"
#include "EventManager.h"

#include "Camera.h"
#include "SceneManager.h"

int EventManager::m_ScrollWheelValue{};

void EventManager::ProcessInputs(bool& isLooping, bool& takeScreenshot, float elapsedSec)
{
	m_ScrollWheelValue = 0;
	SDL_Event e;
	while (SDL_PollEvent(&e) != 0)
	{
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
	SceneManager& sm = *SceneManager::GetInstance();

	switch (e.keysym.sym)
	{
	case::SDLK_TAB:
		sm.ChangeSceneGraph();
		break;
	case::SDLK_MINUS:
		break;
	case::SDLK_r:
		sm.ToggleDepthColour();
		break;
	case::SDLK_f:
		sm.ToggleSampleState();
		break;
	}
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

void EventManager::GetMouseButtonStates(bool& isLmb, bool& isRmb)
{
	int x, y;
	const Uint32 buttons = SDL_GetMouseState(&x, &y);
	isLmb = buttons & SDL_BUTTON_LMASK;
	isRmb = buttons & SDL_BUTTON_RMASK;
}

void EventManager::GetRelativeMouseValues(float& xValue, float& yValue)
{
	int x, y;
	const Uint32 buttons = SDL_GetRelativeMouseState(&x, &y);
	xValue = float(x);
	yValue = float(y);
}

void EventManager::GetScrollWheelValue(int& scrollWheelValue)
{
	scrollWheelValue = m_ScrollWheelValue;
}