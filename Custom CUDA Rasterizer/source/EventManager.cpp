#include "EventManager.h"

#include "Camera.h"
#include "SceneManager.h"

EventManager* EventManager::m_pEventManager{ nullptr };

EventManager::~EventManager()
{
	m_pEventManager = nullptr;
}

void EventManager::ProcessInputs(bool& isLooping, bool& takeScreenshot, float elapsedSec)
{
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
				m_IsRelativeMouse = !m_IsRelativeMouse;
				if (m_IsRelativeMouse)
					SDL_SetRelativeMouseMode(SDL_TRUE);
				else
					SDL_SetRelativeMouseMode(SDL_FALSE);
			}
			KeyDownEvent(e.key);
			break;
		case SDL_KEYUP:
			if (e.key.keysym.scancode == SDL_SCANCODE_X) takeScreenshot = true;
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

	//--------- CameraMovement ---------
	Camera::GetInstance()->ProcessInputs(m_IsLMB, m_IsRMB, m_IsMouse3, elapsedSec);
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
		m_IsLMB = true;
		break;
	case SDL_BUTTON_RIGHT:
		m_IsRMB = true;
		break;
	case SDL_BUTTON_MIDDLE:
		m_IsMouse3 = true;
		break;
	}
}

void EventManager::MouseUpEvent(const SDL_MouseButtonEvent & e)
{
	switch (e.button)
	{
	case SDL_BUTTON_LEFT:
		m_IsLMB = false;
		break;
	case SDL_BUTTON_RIGHT:
		m_IsRMB = false;
		break;
	case SDL_BUTTON_MIDDLE:
		m_IsMouse3 = false;
		break;
	}
}

void EventManager::MouseWheelEvent(const SDL_MouseWheelEvent & e)
{
	Camera& cam = *Camera::GetInstance();

	switch (e.y)
	{
	case 1:
		//Camera::ChangeSpeed(e.y);
		cam.ChangeSpeed(1.f);
		break;
	case -1:
		//Camera::ChangeSpeed(e.y);
		cam.ChangeSpeed(-1.f);
		break;
	}
}

void EventManager::GetRelativeMouseValues(float& xValue, float& yValue)
{
	int x{}, y{};
	//Uint32 mouseButton = SDL_GetRelativeMouseState(&x, &y);
	SDL_GetRelativeMouseState(&x, &y);
	xValue = float(x);
	yValue = float(y);
}