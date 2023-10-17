#pragma once
#include "Math.h"

class Camera
{
public:
	Camera(const FPoint3& position = {}, float fov = 45.f);
	virtual ~Camera() = default;

	void SetPosition(const FPoint3& pos) { m_Position = pos; }

	void TranslateX(float value);
	void TranslateY(float value);
	void TranslateZ(float value);

	void Pitch(float degrees);
	void Yaw(float degrees);

	const FPoint3& GetPosition() const { return m_Position; }
	const FVector3& GetRight() const { return m_Right; }
	const FVector3& GetUp() const { return m_Up; }
	const FVector3& GetForward() const { return m_Forward; }

	const FMatrix4& GetLookAtMatrix() const { return m_LookAtMatrix; }
	const FMatrix4& GetProjectionMatrix() const { return m_ProjectionMatrix; }
	FMatrix4 GetViewMatrix() const { return Inverse(GetLookAtMatrix()); }
	FMatrix3 GetRotationMatrix() const { return FMatrix3{ m_Right, m_Up, m_Forward }; }

	float SetFov(float degrees);
	float GetFov() const { return m_FOV; }

	void GetNearFarValues(float& near, float& far);
	void SetNearFarValues(float near, float far);

	void SetAspectRatio(float width, float height);
	float GetAspectRatio() const { return m_AspectRatio; }

	void SetMoveSpeed(float value);
	float GetMoveSpeed() const { return m_MoveSpeed; }
	void SetRotationSpeed(float value);
	float GetRotationSpeed() const { return m_RotationSpeed; }

protected:
	float m_FOV;
	float m_AspectRatio;
	float m_Near;
	float m_Far;
	float m_MoveSpeed;
	float m_RotationSpeed;

	FPoint3& m_Position;
	FVector3& m_Right;
	FVector3& m_Up;
	FVector3& m_Forward;
	FMatrix4 m_ProjectionMatrix;
	FMatrix4 m_LookAtMatrix;

	void RecalculateProjectionMatrix();
	void RecalculateVectors();

	static const FVector3 m_WorldUp;
};