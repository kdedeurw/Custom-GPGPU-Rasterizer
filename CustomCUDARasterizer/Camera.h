#pragma once
#include "Math.h"

class Camera
{
public:
	explicit Camera(const FPoint3& position, float fov = 45.f);
	virtual ~Camera() = default;

	void SetPos(const FPoint3& pos) { m_Position = pos; };
	void Update(float elapsedSec);

	FPoint3 GetPos() const { return GetPos(GetLookAtMatrix()); };
	FPoint3 GetPos(const FMatrix4& lookatMatrix) const { return FPoint3{ lookatMatrix(0, 3), lookatMatrix(1, 3), lookatMatrix(2, 3) }; };
	const FVector3& GetRight() const { return m_Right; };
	const FVector3& GetUp() const { return m_Up; };
	const FVector3& GetForward() const { return m_Forward; };

	FMatrix4 GetLookAtMatrix() const;
	FMatrix4 GetViewMatrix() const { return Inverse(GetLookAtMatrix()); };
	FMatrix4 GetViewMatrix(const FMatrix4& lookatMatrix) const { return Inverse(lookatMatrix); };
	FMatrix3 GetRotationMatrix() const { return FMatrix3{ m_Right, m_Up, m_Forward }; };
	FMatrix4 GetProjectionMatrix() const;

	float SetFov(float degrees);
	float GetFov() const { return m_FOV; };

	float GetNearValue() const { return m_Near; }
	float GetFarValue() const { return m_Far; }
	//TODO: use dirty flag for that extra bit of spicy performance/optimization
	void SetNearValue(float near);
	void SetFarValue(float far);

	void RotateVector(FVector3& direction);

	void SetAspectRatio(float width, float height);
	float GetAspectRatio() const { return m_AspectRatio; };

	void ChangeSpeed(float value);

private:
	float m_FOV;
	float m_AspectRatio;
	float m_CameraSpeed;
	float m_Near;
	float m_Far;

	FPoint3 m_Position;
	FVector3 m_Forward;
	FVector3 m_Up;
	FVector3 m_Right;

	static const FVector3 m_WorldUp;

public:
	void RecalculateRightUpVectors();

	void TranslateX(float value);
	void TranslateY(float value);
	void TranslateZ(float value);

	void PrintCamSpeed();
};