#include "PCH.h"
#include "Camera.h"
#include "EventManager.h"
#include <iostream>

const FVector3 Camera::m_WorldUp{ 0.f, 1.f, 0.f };

Camera::Camera(const FPoint3& position, float fov)
	: m_FOV{ SetFov(fov) }
	, m_AspectRatio{}
	, m_CameraSpeed{ 10.f }
	, m_Near{ 0.1f }
	, m_Far{ 100.f }
	, m_Position{ position }
	, m_Forward{ 0, 0, 1.f }
	, m_Up{ 0, 1.f, 0 }
	, m_Right{ 1.f, 0, 0 }
{
}

float Camera::SetFov(float degrees)
{
	m_FOV = tanf(ToRadians(degrees) / 2.f);
	return m_FOV;
}

void Camera::SetAspectRatio(float width, float height)
{
	m_AspectRatio = width / height;
}

FMatrix4 Camera::GetLookAtMatrix() const
{
	return FMatrix4{
		m_Right.x, m_Up.x, m_Forward.x, m_Position.x,
		m_Right.y, m_Up.y, m_Forward.y, m_Position.y,
		m_Right.z, m_Up.z, m_Forward.z, m_Position.z,
		0.f, 0.f, 0.f, 1.f  };
}

FMatrix4 Camera::GetProjectionMatrix() const
{
	const float A{ m_Far / (m_Near - m_Far) };
	const float B{ ((m_Far * m_Near) / (m_Near - m_Far)) };
	const FMatrix4 projectionMatrix = 
	{	1.f / (m_AspectRatio * m_FOV), 0.f, 0.f, 0.f,
		0.f, 1.f / m_FOV, 0.f, 0.f,
		0.f, 0.f, A, B,
		0.f, 0.f, -1.f, 0.f };
	// negative Z & W values bc right-handed coordinate system!
	return projectionMatrix;
}

void Camera::RecalculateRightUpVectors()
{
	Normalize(m_Right);
	m_Right = Cross(m_WorldUp, m_Forward);
	Normalize(m_Up);
	m_Up = Cross(m_Forward, m_Right);
}

void Camera::SetNearValue(float near)
{
	m_Near = near;
}
void Camera::SetFarValue(float far)
{
	m_Far = far;
}

void Camera::RotateVector(FVector3& direction)
{
	direction = GetRotationMatrix() * direction;
}

void Camera::TranslateX(float value)
{
	m_Position += m_Right * value;
}

void Camera::TranslateY(float value)
{
	m_Position += m_WorldUp * value;
}

void Camera::TranslateZ(float value)
{
	m_Position += m_Forward * value;
}

void Camera::Update(float elapsedSec)
{
	const Uint8* pStates = SDL_GetKeyboardState(nullptr);
	const float limit{ 250.f };

	float x{}, y{};
	EventManager::GetRelativeMouseValues(x, y);

	bool isLmb{}, isRmb{};
	EventManager::GetMouseButtonStates(isLmb, isRmb);

	int scrollWheelValue{};
	EventManager::GetScrollWheelValue(scrollWheelValue);
	if (scrollWheelValue != 0)
	{
		ChangeSpeed((float)scrollWheelValue);
	}

	bool hasMoved{}, hasRotated{};
	if (isLmb && isRmb)
	{
		TranslateY(y / 50.f);
		hasMoved = true;
	}
	else if (isLmb && !isRmb)
	{
		TranslateZ(y / 50.f);
		m_Forward = MakeRotationY(float(x) / limit) * m_Forward;
		//m_Forward = MakeRotation(float(y / limit)) * m_Forward;
		hasRotated = true;
	}
	else if (isRmb && !isLmb)
	{
		m_Forward = MakeRotationZYX(float(-y / limit), float(-x / limit), 0.f) * m_Forward;
		hasRotated = true;
	}

	if (pStates[SDL_SCANCODE_W])
	{
		TranslateZ(-m_CameraSpeed * elapsedSec);
		hasMoved = true;
	}
	else if (pStates[SDL_SCANCODE_S])
	{
		TranslateZ(m_CameraSpeed * elapsedSec);
		hasMoved = true;
	}
	if (pStates[SDL_SCANCODE_A])
	{
		TranslateX(-m_CameraSpeed * elapsedSec);
		hasMoved = true;
	}
	else if (pStates[SDL_SCANCODE_D])
	{
		TranslateX(m_CameraSpeed * elapsedSec);
		hasMoved = true;
	}
	
	//Positional changes directly impact the lookat matrix's pos
	if (hasRotated)
	{
		RecalculateRightUpVectors();
		Normalize(m_Forward);
	}
}

void Camera::ChangeSpeed(float value)
{
	m_CameraSpeed += value;
	if (m_CameraSpeed < 1.f || m_CameraSpeed > 25.f)
	{
		m_CameraSpeed -= value;
		std::cout << "\n!Exceeded speed!\n";
	}
	PrintCamSpeed();
}

void Camera::PrintCamSpeed()
{
	std::cout << "Cam Speed: " << m_CameraSpeed << '\n';
}