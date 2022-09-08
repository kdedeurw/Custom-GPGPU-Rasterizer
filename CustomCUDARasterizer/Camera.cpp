#include "PCH.h"
#include "Camera.h"

const FVector3 Camera::m_WorldUp{ 0.f, 1.f, 0.f };

Camera::Camera(const FPoint3& position, float fov)
	: m_FOV{ SetFov(fov) }
	, m_AspectRatio{}
	, m_Near{ 0.1f }
	, m_Far{ 100.f }
	, m_MoveSpeed{ 5.f }
	, m_RotationSpeed{ 1.f }
	, m_Position{ reinterpret_cast<FPoint3&>(m_LookAtMatrix[3][0]) }
	, m_Right{ reinterpret_cast<FVector3&>(m_LookAtMatrix[0][0]) }
	, m_Up{ reinterpret_cast<FVector3&>(m_LookAtMatrix[1][0]) }
	, m_Forward{ reinterpret_cast<FVector3&>(m_LookAtMatrix[2][0]) }
	, m_ProjectionMatrix{ FMatrix4::Identity() }
	, m_LookAtMatrix{ FMatrix4::Identity() }
{
	m_Position = position;
	//m_Right = { 1.f, 0.f, 0.f };
	//m_Up = { 0.f, 1.f, 0.f };
	//m_Forward = { 0.f, 0.f, 1.f };
}

float Camera::SetFov(float degrees)
{
	const float radians = ToRadians(degrees);
	m_FOV = tanf(radians / 2.f);
	return m_FOV;
}

void Camera::SetAspectRatio(float width, float height)
{
	m_AspectRatio = width / height;
	RecalculateProjectionMatrix();
}

void Camera::GetNearFarValues(float& near, float& far)
{
	near = m_Near;
	far = m_Far;
}

void Camera::SetNearFarValues(float near, float far)
{
	m_Near = near;
	m_Far = far;
	RecalculateProjectionMatrix();
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

void Camera::Pitch(float degrees)
{
	const float radians = ToRadians(degrees);
	const FMatrix3 rotX = MakeRotationX(radians);
	m_Forward = rotX * m_Forward;
	RecalculateVectors();
}

void Camera::Yaw(float degrees)
{
	const float radians = ToRadians(degrees);
	const FMatrix3 rotY = MakeRotationY(radians);
	m_Forward = rotY * m_Forward;
	RecalculateVectors();
}

void Camera::SetMoveSpeed(float value)
{
	constexpr float minSpeed = 1.f;
	constexpr float maxSpeed = 25.f;
	m_MoveSpeed = Clamp(m_MoveSpeed + value, minSpeed, maxSpeed);
}

void Camera::SetRotationSpeed(float value)
{
	constexpr float minSpeed = 0.1f;
	constexpr float maxSpeed = 5.f;
	m_RotationSpeed = Clamp(m_RotationSpeed + value, minSpeed, maxSpeed);
}

void Camera::RecalculateProjectionMatrix()
{
	const float A = m_Far / (m_Near - m_Far);
	const float B = (m_Far * m_Near) / (m_Near - m_Far);
	const float value = 1.f / (m_AspectRatio * m_FOV);
	const float invFov = 1.f / m_FOV;
	m_ProjectionMatrix[0][0] = value;
	m_ProjectionMatrix[1][1] = invFov;
	m_ProjectionMatrix[2][2] = A;
	m_ProjectionMatrix[2][3] = -1.f; // negative Z & W values bc right-handed coordinate system!
	m_ProjectionMatrix[3][2] = B;
}

void Camera::RecalculateVectors()
{
	Normalize(m_Forward);
	m_Right = Cross(m_WorldUp, m_Forward);
	Normalize(m_Right);
	m_Up = Cross(m_Forward, m_Right);
	Normalize(m_Up);
}