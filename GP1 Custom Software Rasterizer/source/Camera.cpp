#include "Camera.h"
#include "EventManager.h"
#include <iostream>

Camera* Camera::m_pCamera{ nullptr };

Camera::Camera(const FPoint3& position, float fov)
	: m_Position{ position }
	, m_FOV{ tanf(ToRadians(fov) / 2) }
	, m_AspectRatio{}
	, m_Forward{ 0, 0, 1.f }
	, m_Up{ 0, 1.f, 0 }
	, m_Right{ 1.f, 0, 0 }
	, m_LookAt{ FVector4{m_Right, 0}, FVector4{m_Up, 0}, FVector4{m_Forward, 0}, FVector4{0, 0, 0, 1.f} }
	, m_RotationMatrix{}
	, m_Near{ 0.1f }
	, m_Far{ 100.f }
{
}

Camera::~Camera()
{
	m_pCamera = nullptr;
}

void Camera::SetAspectRatio(float width, float height)
{
	m_AspectRatio = width / height;
}

float Camera::GetAspectRatio() const
{
	return m_AspectRatio;
}

float Camera::GetFov() const
{
	return m_FOV;
}

const FPoint3& Camera::GetPos() const
{
	return m_Position;
}

FMatrix4 Camera::GetProjectionMatrix() const
{
	// projectionmatrix
	float A{ m_Far / (m_Near - m_Far) };
	float B{ ((m_Far * m_Near) / (m_Near - m_Far)) };
	FMatrix4 projectionMatrix{
		1.f / (m_AspectRatio * m_FOV), 0.f, 0.f, 0.f,
		0.f, 1.f / m_FOV, 0.f, 0.f,
		0.f, 0.f, A, B,
		0.f, 0.f, -1.f, 0.f
	};
	// negative Z & W values bc right-handed coordinate system!
	return projectionMatrix;
}

const FMatrix4& Camera::GenerateLookAt()
{
	FVector4 right{ GetRight() };
	FVector4 up{ GetUp() };
	m_LookAt = FMatrix4{
		right.x, up.x, m_Forward.x, m_Position.x,
		right.y, up.y, m_Forward.y, m_Position.y,
		right.z, up.z, m_Forward.z, m_Position.z,
		0.f, 0.f, 0.f, 1.f };
	//m_LookAt = FMatrix4{ GetRight(), GetUp(), m_Forward, FVector4{FVector3{m_Position}, 1.f} };
	return m_LookAt;
}

const FMatrix3& Camera::GenerateRotation()
{
	m_RotationMatrix = FMatrix3(GetRight(), GetUp(), m_Forward);
	return m_RotationMatrix;
}

void Camera::Rotate(FVector3& direction)
{
	GenerateRotation();
	direction = m_RotationMatrix * direction;
}

const FMatrix4 Camera::GetLookAtMatrixConst() const
{
	return FMatrix4{ FVector4{m_Right, 0}, FVector4{m_Up, 0}, FVector4{m_Forward, 0}, FVector4{FVector3{m_Position}, 1.f} };
}

const FVector3& Camera::GetRight()
{
	m_Right = Elite::Cross(m_WorldUp, m_Forward);
	//Elite::Normalize(m_Right);
	return m_Right;
}

const FVector3& Camera::GetUp()
{
	m_Up = Elite::Cross(m_Forward, m_Right);
	//Elite::Normalize(m_Up);
	return m_Up;
}

const FVector3& Camera::GetForward(const FPoint3& origin)
{
	m_Forward = { m_Position - origin };
	//Elite::Normalize(m_Forward);
	return m_Forward;
}

void Camera::TranslateX(float value)
{
	m_Position += m_Right * value;
}

void Camera::TranslateY(float value)
{
	//m_Position += m_Up * value;
	m_Position += m_WorldUp * value;
}

void Camera::TranslateZ(float value)
{
	m_Position += m_Forward * value;
	//m_Position.z += MakeTranslation(m_Forward).data[3][2];
}

void Camera::ProcessInputs(bool lmb, bool rmb, bool mouse3, float elapsedSec)
{
	const float limit{ 250.f };
	float x{}, y{};
	EventManager::GetInstance()->GetRelativeMouseValues(x, y);
	const Uint8* pStates = SDL_GetKeyboardState(nullptr);

	if (lmb && rmb)
	{
		TranslateY(y / 50.f);
	}
	else if (lmb && !rmb)
	{
		TranslateZ(y / 50.f);
		m_Forward = MakeRotationY(float(x) / limit) * m_Forward;
		//m_Forward = MakeRotation(float(y / limit)) * m_Forward;
	}
	else if (rmb && !lmb)
	{
		//FMatrix4 temp = MakeTranslation(m_Forward) * m_Position;
		//m_Position.x += MakeTranslation(m_Forward).data[3][0];
		//MakeRotation(x);
		//m_RotationMatrix;
		//MakeRotation(m_RotationMatrix);
		m_Forward = MakeRotationZYX(float(-y / limit), float(-x / limit), 0.f) * m_Forward;

		// !ROTATING AROUND FIXED POSITION!, seems like

		//MakeTranslation(FVector2{m_Position.x, m_Position.y}, );
	}

	if (pStates[SDL_SCANCODE_W])
	{
		TranslateZ(-m_CameraSpeed * elapsedSec);
	}
	else if (pStates[SDL_SCANCODE_S])
	{
		TranslateZ(m_CameraSpeed * elapsedSec);
	}
	if (pStates[SDL_SCANCODE_A])
	{
		TranslateX(-m_CameraSpeed * elapsedSec);
	}
	else if (pStates[SDL_SCANCODE_D])
	{
		TranslateX(m_CameraSpeed * elapsedSec);
	}

	GenerateLookAt();
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