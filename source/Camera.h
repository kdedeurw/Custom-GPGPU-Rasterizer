#pragma once
#include "EMath.h"
#include "EMathUtilities.h"

using namespace Elite;

class Camera
{
public:
	static void CreateInstance(const FPoint3& position, float fov)
	{
		if (!m_pCamera) m_pCamera = new Camera{ position, fov };
	}
	static Camera* GetInstance()
	{
		if (!m_pCamera) return nullptr;
		return m_pCamera;
	}
	~Camera();

	const FVector3& GetRight();
	const FVector3& GetUp();
	const FVector3& GetForward(const FPoint3& origin);

	const FMatrix4& GenerateLookAt();
	const FMatrix3& GenerateRotation();
	const FMatrix4 GetLookAtMatrixConst() const;

	float GetFov() const;
	const FPoint3& GetPos() const;

	void Rotate(FVector3& direction);

	void SetAspectRatio(float width, float height);
	float GetAspectRatio() const;

	void ProcessInputs(bool lmb, bool rmb, bool mouse3, float elapsedSec);
	void ChangeSpeed(float value);

	FMatrix4 GetProjectionMatrix() const;

private:
	explicit Camera(const FPoint3& position, float fov);

	static Camera* m_pCamera;

	float m_FOV;
	float m_AspectRatio;
	float m_CameraSpeed = 10.f;

	float m_Near;
	float m_Far;

	FPoint3 m_Position;

	FMatrix3 m_RotationMatrix;

	FVector3 m_Forward = { 0.f, 0.f, 1.f };
	const FVector3 m_WorldUp = { 0.f, 1.f, 0.f };
	FVector3 m_Up = { 0.f, 1.f, 0.f };
	FVector3 m_Right = { 1.f, 0.f, 0.f };
	FMatrix4 m_LookAt = { FVector4{m_Right, 0}, FVector4{m_Up, 0}, FVector4{m_Forward, 0}, FVector4{0, 0, 0, 1.f} };

	void TranslateX(float value);
	void TranslateY(float value);
	void TranslateZ(float value);

	void PrintCamSpeed();
};

